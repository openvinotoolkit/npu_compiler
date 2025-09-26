//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedBufferType = !VPUIP.DistributedBuffer<1x896x288x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]], memory_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]]}>
!OverlappedBufferType = !VPUIP.DistributedBuffer<1x896x288x4xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]], memory_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]]}>

// CHECK-LABEL: func.func @DontOptimizeForInplaceUser
func.func @DontOptimizeForInplaceUser(%arg0 : !DistributedBufferType, %arg1 : !DistributedBufferType, %arg2 : !OverlappedBufferType) -> !OverlappedBufferType {
    %alloc = memref.alloc() : memref<1x1792x288x4xf16, #NHWC, @DDR>
    %subview0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 896, 288, 4] : memref<1x1792x288x4xf16, #NHWC, @DDR> to memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : !DistributedBufferType) outputs(%subview0 : memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) -> memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>

    %subview1 = VPUIP.SubView %alloc [0, 896, 0, 0] [1, 896, 288, 4] : memref<1x1792x288x4xf16, #NHWC, @DDR> to memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : !DistributedBufferType) outputs(%subview1 : memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) -> memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy0, %copy1: memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>, memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) outputs(%alloc : memref<1x1792x288x4xf16, #NHWC, @DDR>) -> memref<1x1792x288x4xf16, #NHWC, @DDR>

    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 896, 288, 4] : memref<1x1792x288x4xf16, #NHWC, @DDR> to memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>
    %allocDistributed = VPURT.AllocDistributed -> !OverlappedBufferType

    %copy2 = VPUIP.Copy inputs(%subview2 : memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) outputs(%allocDistributed : !OverlappedBufferType) -> !OverlappedBufferType

    %nceClusterTask = VPUIP.NCEClusterTask {eltwise_type = #VPU.eltwise_type<MULTIPLY>, is_inplace = true, minimumHardwareExecutionCost = 37893 : i64, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%copy2 : !OverlappedBufferType)
            weights(%arg2 : !OverlappedBufferType)
            parent_input(%copy2 : !OverlappedBufferType)
            parent_output(%allocDistributed : !OverlappedBufferType)
            outputs(%allocDistributed : !OverlappedBufferType) -> !OverlappedBufferType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [3, 95, 895], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [3, 95, 895], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [3, 95, 895], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [3, 95, 895], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [3, 95, 895], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [3, 95, 895], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
    }

    return %nceClusterTask : !OverlappedBufferType

    // Check the concat pattern is unchange because the buffer is shared by inplace eltwise's output
    // CHECK:       [[ALLOC:%.+]] = memref.alloc()
    // CHECK:       [[SUBVIEW0:%.+]] = VPUIP.SubView [[ALLOC]]
    // CHECK:       [[COPY0:%.+]] = VPUIP.Copy
    // CHECK:       [[SUBVIEW1:%.+]] = VPUIP.SubView [[ALLOC]]
    // CHECK:       [[COPY1:%.+]] = VPUIP.Copy
    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[COPY0]], [[COPY1]]
    // CHECK:       [[SUBVIEW2:%.+]] = VPUIP.SubView [[CONCAT]]
    // CHECK:       [[ALLOC_DISTRIBUTED:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[SUBVIEW2]]
    // CHECK:       [[NCE_CLUSTER_TASK:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[COPY2]]
    // CHECK-SAME:      outputs([[ALLOC_DISTRIBUTED]]
    // CHECK:       return [[NCE_CLUSTER_TASK]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed0 = !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Distributed1 = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>
!Arg1T = memref<1x32x128x1xf32, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedOverlappedWithConvertDMARightBranchFromDDR
// CHECK-SAME:  ([[LEFT_INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>, [[RIGHT_INPUT_ARG:%.+]]: memref<1x32x128x1xf32, @DDR>)
func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedOverlappedWithConvertDMARightBranchFromDDR(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPURT.AllocDistributed -> !Distributed0
    %3 = VPUIP.Copy inputs(%arg1 : memref<1x32x128x1xf32, @DDR>) outputs(%2 : !Distributed0) -> !Distributed0
    %4 = VPURT.AllocDistributed -> !Distributed1
    %5 = VPUIP.ConvertDMA inputs(%3 : !Distributed0) outputs(%4 : !Distributed1) -> !Distributed1
    %6 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %7 = VPUIP.Copy inputs(%5 : !Distributed1) outputs(%6 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %8 = VPUIP.ConcatView
        inputs(%1, %7 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %9 = VPUIP.GenericReshape inputs(%8 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %10 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%9 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %11 = VPUIP.SubView %10 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %12 = VPUIP.SubView %10 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %13 = VPURT.AllocDistributed -> !ResultT
    %14 = VPUIP.Copy inputs(%11 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%13 : !ResultT) -> !ResultT
    %15 = VPURT.AllocDistributed -> !ResultT
    %16 = VPUIP.Copy inputs(%12 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%15 : !ResultT) -> !ResultT

    return %14, %16 : !ResultT, !ResultT

    // CHECK:                   [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[RIGHT_INPUT_ARG]] : memref<1x32x128x1xf32, @DDR>) -> memref<4096x1x1x1xf32, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf32, @DDR>) -> memref<4096x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_0]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_0:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_2]] : memref<128x1x1x1xf32, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_0]], [[CONVERTDMA_0]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>)
    // CHECK-SAME:                    outputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_1:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_6]] : memref<128x1x1x1xf32, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[CONVERTDMA_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>)
    // CHECK-SAME:                    outputs([[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!ResultT = !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]
}>

!Distributed0 = !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Distributed1 = !VPUIP.DistributedBuffer<1x32x128x1xf16, #NCHW, @CMX_NN, {
    mode = "DUPLICATED",
    num_tiles = [1, 1, 4, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]
}>

!Arg0T = memref<1x32x128x1023xf16, @DDR>
!Arg1T = memref<1x32x128x1xf32, @DDR>

// CHECK-LABEL: func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedDuplicatedWithConvertDMARightBranchFromCMX
// CHECK-SAME:  ([[INPUT_ARG:%.+]]: memref<1x32x128x1023xf16, @DDR>)
func.func @SplitUnbalancedConcatOnDifferentAxisBranchInputIsDistributedDuplicatedWithConvertDMARightBranchFromCMX(%arg0 : !Arg0T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x128x1024xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 128, 1023] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x128x1023xf16, @DDR>) outputs(%0 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    // Right branch
    %2 = VPURT.AllocDistributed -> !Distributed0
    %3 = VPURT.AllocDistributed -> !Distributed1
    %4 = VPUIP.ConvertDMA inputs(%2 : !Distributed0) outputs(%3 : !Distributed1) -> !Distributed1
    %5 = VPUIP.SubView %alloc [0, 0, 0, 1023] [1, 32, 128, 1] : memref<1x32x128x1024xf16, @DDR> to memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : !Distributed1) outputs(%5 : memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>) -> memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>
    %7 = VPUIP.ConcatView
        inputs(%1, %6 : memref<1x32x128x1023xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>, memref<1x32x128x1xf16, {order = #NCHW, strides = [4194304, 131072, 1024, 1]}, @DDR>)
        outputs(%alloc : memref<1x32x128x1024xf16, @DDR>) -> memref<1x32x128x1024xf16, @DDR>
    %8 = VPUIP.GenericReshape inputs(%7 : memref<1x32x128x1024xf16, @DDR>) -> memref<4096x1024x1x1xf16, @DDR>
    %9 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%8 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, #NHWC, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %11 = VPUIP.SubView %9 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, #NHWC, @DDR> to memref<128x1024x1x1xf16, #NHWC, @DDR>
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%10 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT
    %14 = VPURT.AllocDistributed -> !ResultT
    %15 = VPUIP.Copy inputs(%11 : memref<128x1024x1x1xf16, #NHWC, @DDR>) outputs(%14 : !ResultT) -> !ResultT

    return %13, %15 : !ResultT, !ResultT

    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>
    // CHECK:                   [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1xf32, @DDR>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>) outputs(
    // CHECK:                   [[ALLOC]] : memref<1x32x128x1xf32, @DDR>) -> memref<1x32x128x1xf32, @DDR>
    // CHECK:                   [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x128x1xf32, @DDR>) -> memref<4096x1x1x1xf32, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf32, @DDR>) -> memref<4096x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_0:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_2]] : memref<128x1x1x1xf32, #NHWC, @DDR>) outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[CONVERTDMA_0]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:                   [[DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[DISTRIBUTED_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, #NHWC, @DDR> to memref<128x1023x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, #NHWC, @DDR>) outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, #NHWC, @DDR> to memref<128x1x1x1xf32, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_1:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_6]] : memref<128x1x1x1xf32, #NHWC, @DDR>) outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_2]], [[CONVERTDMA_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>, !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) outputs(
    // CHECK:                   [[DISTRIBUTED_2]] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   return [[CONCATVIEW_0]], [[CONCATVIEW_1]]
}

//
// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ResultT = !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [4, 1, 1, 1],
    num_clusters = 4 : i64,
    alignment = [16, 1, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]
}>

!Arg0T = memref<1x32x768x96xf16, @DDR>
!Arg1T = !VPUIP.DistributedBuffer<1x32x256x96xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 4, 1, 1],
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96]],
    compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    memory_shapes = [[1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96]],
    memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]
}>

// CHECK-LABEL: func.func @NotSplitUnbalancedDDRConcatOnSameAxisWhenNoLeftBranchDataOnTheLastCluster
// CHECK-SAME:  [[ARG0:%.+]]: memref<1x32x768x96xf16, @DDR>,
// CHECK-SAME:  [[ARG1:%.+]]: !VPUIP.DistributedBuffer<1x32x256x96xf16, #NCHW, @CMX_NN
func.func @NotSplitUnbalancedDDRConcatOnSameAxisWhenNoLeftBranchDataOnTheLastCluster(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!ResultT, !ResultT) {
    %alloc = memref.alloc() : memref<1x32x1024x96xf16, @DDR>
    // Left branch
    %0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 32, 768, 96] : memref<1x32x1024x96xf16, @DDR> to memref<1x32x768x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %1 = VPUIP.Copy inputs(%arg0 : memref<1x32x768x96xf16, @DDR>) outputs(%0 : memref<1x32x768x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>) -> memref<1x32x768x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>

    // Right branch
    %alloc_1 = memref.alloc() : memref<1x32x256x96xf16, @DDR>
    %2 = VPUIP.Copy inputs(%arg1 : !Arg1T) outputs(%alloc_1 : memref<1x32x256x96xf16, @DDR>) -> memref<1x32x256x96xf16, @DDR>

    %3 = VPUIP.SubView %alloc [0, 0, 768, 0] [1, 32, 256, 96] : memref<1x32x1024x96xf16, @DDR> to memref<1x32x256x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x32x256x96xf16, @DDR>) outputs(%3 : memref<1x32x256x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>) -> memref<1x32x256x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>
    %5 = VPUIP.ConcatView inputs(%1, %4 : memref<1x32x768x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>, memref<1x32x256x96xf16, {order = #NCHW, strides = [3145728, 98304, 96, 1]}, @DDR>) outputs(%alloc : memref<1x32x1024x96xf16, @DDR>) -> memref<1x32x1024x96xf16, @DDR>

    %6 = VPUIP.GenericReshape inputs(%5 : memref<1x32x1024x96xf16, @DDR>) -> memref<32768x96x1x1xf16, @DDR>
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<32768x96x1x1xf16, @DDR>) -> memref<32768x96x1x1xf16, #NHWC, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1024, 96, 1, 1] : memref<32768x96x1x1xf16, #NHWC, @DDR> to memref<1024x96x1x1xf16, #NHWC, @DDR>
    %9 = VPUIP.SubView %7 [1024, 0, 0, 0] [1024, 96, 1, 1] : memref<32768x96x1x1xf16, #NHWC, @DDR> to memref<1024x96x1x1xf16, #NHWC, @DDR>

    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<1024x96x1x1xf16, #NHWC, @DDR>) outputs(%10 : !ResultT) -> !ResultT

    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<1024x96x1x1xf16, #NHWC, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    return %11, %13 : !ResultT, !ResultT

    // CHECK:                   [[ALLOC:%.+]] = memref.alloc() : memref<1x32x256x96xf16, @DDR>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[ARG1]] : !VPUIP.DistributedBuffer<1x32x256x96xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:                    outputs([[ALLOC:%.+]] : memref<1x32x256x96xf16, @DDR>) -> memref<1x32x256x96xf16, @DDR>

    // CHECK:                   [[RESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x32x768x96xf16, @DDR>) -> memref<24576x96x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE_0]] : memref<24576x96x1x1xf16, @DDR>) -> memref<24576x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[RESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x256x96xf16, @DDR>) -> memref<8192x96x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE_1]] : memref<8192x96x1x1xf16, @DDR>) -> memref<8192x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[ALLOC_DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   [[ALLOC_0:%.+]] = memref.alloc() : memref<1024x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [768, 96, 1, 1] : memref<24576x96x1x1xf16, #NHWC, @DDR> to memref<768x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ALLOC_0:%.+]] [0, 0, 0, 0] [768, 96, 1, 1] : memref<1024x96x1x1xf16, #NHWC, @DDR> to memref<768x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<768x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_1]] : memref<768x96x1x1xf16, #NHWC, @DDR>) -> memref<768x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [256, 96, 1, 1] : memref<8192x96x1x1xf16, #NHWC, @DDR> to memref<256x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[ALLOC_0]] [768, 0, 0, 0] [256, 96, 1, 1] : memref<1024x96x1x1xf16, #NHWC, @DDR> to memref<256x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<256x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_3]] : memref<256x96x1x1xf16, #NHWC, @DDR>) -> memref<256x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]] : memref<768x96x1x1xf16, #NHWC, @DDR>, memref<256x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_0]] : memref<1024x96x1x1xf16, #NHWC, @DDR>) -> memref<1024x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[COPY_3:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_0]] : memref<1024x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   [[ALLOC_DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   [[ALLOC_1:%.+]] = memref.alloc() : memref<1024x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [768, 0, 0, 0] [768, 96, 1, 1] : memref<24576x96x1x1xf16, #NHWC, @DDR> to memref<768x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[ALLOC_1:%.+]] [0, 0, 0, 0] [768, 96, 1, 1] : memref<1024x96x1x1xf16, #NHWC, @DDR> to memref<768x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<768x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_5]] : memref<768x96x1x1xf16, #NHWC, @DDR>) -> memref<768x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [256, 0, 0, 0] [256, 96, 1, 1] : memref<8192x96x1x1xf16, #NHWC, @DDR> to memref<256x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[ALLOC_1]] [768, 0, 0, 0] [256, 96, 1, 1] : memref<1024x96x1x1xf16, #NHWC, @DDR> to memref<256x96x1x1xf16, #NHWC, @DDR>
    // CHECK:                   [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<256x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_7]] : memref<256x96x1x1xf16, #NHWC, @DDR>) -> memref<256x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]] : memref<768x96x1x1xf16, #NHWC, @DDR>, memref<256x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_1]] : memref<1024x96x1x1xf16, #NHWC, @DDR>) -> memref<1024x96x1x1xf16, #NHWC, @DDR>

    // CHECK:                   [[COPY_6:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_1]] : memref<1024x96x1x1xf16, #NHWC, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_DISTRIBUTED_1]] : !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   return [[COPY_3]], [[COPY_6]]
}


// -----
//

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!ResultT = !VPUIP.DistributedBuffer<1x64x40x160xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 14, 160], [1, 64, 13, 160], [1, 64, 13, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0], [0, 0, 27, 0]],
    memory_shapes = [[1, 64, 14, 160], [1, 64, 13, 160], [1, 64, 13, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0], [0, 0, 27, 0]]
}>

!DistType1 = !VPUIP.DistributedBuffer<1x40x1x64xf32, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 14, 1, 64], [1, 13, 1, 64], [1, 13, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 14, 0, 0], [0, 27, 0, 0]],
    memory_shapes = [[1, 14, 1, 64], [1, 13, 1, 64], [1, 13, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 14, 0, 0], [0, 27, 0, 0]]
}>

!DistType2 = !VPUIP.DistributedBuffer<1x40x1x64xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 14, 1, 64], [1, 13, 1, 64], [1, 13, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 14, 0, 0], [0, 27, 0, 0]],
    memory_shapes = [[1, 14, 1, 64], [1, 13, 1, 64], [1, 13, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 14, 0, 0], [0, 27, 0, 0]]
}>

!DistType3 = !VPUIP.DistributedBuffer<1x40x1x64xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 14, 1, 64], [1, 13, 1, 64], [1, 13, 1, 64]],
    compute_offsets = [[0, 0, 0, 0], [0, 14, 0, 0], [0, 27, 0, 0]],
    memory_shapes = [[1, 14, 1, 64], [1, 13, 1, 64], [1, 13, 1, 64]],
    memory_offsets = [[0, 0, 0, 0], [0, 14, 0, 0], [0, 27, 0, 0]]
}>

!DistType4 = !VPUIP.DistributedBuffer<1x64x40x160xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @CMX_NN, {
    mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 64, 14, 160], [1, 64, 13, 160], [1, 64, 13, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 14, 0], [0, 0, 27, 0]],
    memory_shapes = [[1, 64, 14, 160], [1, 64, 13, 160], [1, 64, 13, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 14, 0], [0, 0, 27, 0]]
}>

!Arg0T = memref<1x40x639x64xf16, @DDR>
!Arg1T = memref<1x40x1x128xf32, @DDR>


// CHECK-LABEL: func.func @NotSplitUnbalancedConcatOnDifferentAxisForStrideInput
// CHECK-SAME: ([[LEFT_INPUT_ARG:%.+]]: memref<1x40x639x64xf16, @DDR>,
// CHECK-SAME: [[RIGHT_INPUT_ARG:%.+]]: memref<1x40x1x128xf32, @DDR>
func.func @NotSplitUnbalancedConcatOnDifferentAxisForStrideInput(%arg0 : !Arg0T, %arg1 : !Arg1T) -> (!ResultT) {
    %alloc = memref.alloc() : memref<1x64x40x640xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, @DDR>
    %alloc_0 = memref.alloc() : memref<1x40x640x64xf16, @DDR>
    %0 = VPUIP.SubView %arg1 [0, 0, 0, 64] [1, 40, 1, 64]
         : memref<1x40x1x128xf32, @DDR>
         to memref<1x40x1x64xf32, {order = #NCHW, strides = [5120, 128, 128, 1]}, @DDR>
    %1 = VPURT.AllocDistributed -> !DistType1
    %2 = VPUIP.Copy inputs(%0 : memref<1x40x1x64xf32, {order = #NCHW, strides = [5120, 128, 128, 1]}, @DDR>)
                  outputs(%1 : !DistType1)
         -> !DistType1
    %3 = VPURT.AllocDistributed -> !DistType2

    %4 = VPUIP.ConvertDMA inputs(%2 : !DistType1)
                        outputs(%3 : !DistType2)
         -> !DistType3
    %5 = VPUIP.SubView %alloc_0 [0, 0, 639, 0] [1, 40, 1, 64]
         : memref<1x40x640x64xf16, @DDR>
         to memref<1x40x1x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>
    %6 = VPUIP.Copy inputs(%4 : !DistType3)
                  outputs(%5 : memref<1x40x1x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>)
         -> memref<1x40x1x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>
    %7 = VPUIP.SubView %alloc_0 [0, 0, 0, 0] [1, 40, 639, 64]
         : memref<1x40x640x64xf16, @DDR>
         to memref<1x40x639x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>
    %8 = VPUIP.Copy inputs(%arg0 : memref<1x40x639x64xf16, @DDR>)
                  outputs(%7 : memref<1x40x639x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>)
         -> memref<1x40x639x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>
    %9 = VPUIP.ConcatView inputs(%8, %6 : memref<1x40x639x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>,
                                   memref<1x40x1x64xf16, {order = #NCHW, strides = [1638400, 40960, 64, 1]}, @DDR>)
                        outputs(%alloc_0 : memref<1x40x640x64xf16, @DDR>)
         -> memref<1x40x640x64xf16, @DDR>
    %10 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW}
          inputs(%9 : memref<1x40x640x64xf16, @DDR>)
          -> memref<1x64x40x640xf16, #NHWC, @DDR>
    %11 = VPUIP.SubView %10 [0, 0, 0, 480] [1, 64, 40, 160]
          : memref<1x64x40x640xf16, #NHWC, @DDR>
          to memref<1x64x40x160xf16, {order = #NHWC, strides = [1638400, 1, 40960, 64]}, @DDR>
    %12 = VPURT.AllocDistributed -> !DistType4
    %13 = VPUIP.Copy inputs(%11 : memref<1x64x40x160xf16, {order = #NHWC, strides = [1638400, 1, 40960, 64]}, @DDR>)
                   outputs(%12 : !DistType4)
          -> !DistType4

    return %13 : !ResultT


    // CHECK:       [[CONCAT_VIEW:%.+]] = VPUIP.ConcatView
    // CHECK:       [[PERMUTE_CAST:%.+]] = VPUIP.PermuteCast
    // CHECK:       [[SUBVIEW:%.+]] = VPUIP.SubView
    // CHECK:       [[ALLOC:%.+]] = VPURT.AllocDistributed
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK:       return [[COPY]]
}
