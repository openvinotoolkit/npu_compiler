//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-concat-view-copies %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedBufferType = !VPUIP.DistributedBuffer<1x896x288x4xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]], memory_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]]}>
!OverlappedBufferType = !VPUIP.DistributedBuffer<1x896x288x4xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments, compute_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]], memory_shapes = [[1, 896, 96, 4], [1, 896, 96, 4], [1, 896, 96, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 96, 0], [0, 0, 192, 0]]}>

// CHECK-LABEL: func.func @DontOptimizeForInplaceUser
func.func @DontOptimizeForInplaceUser(%arg0 : !DistributedBufferType, %arg1 : !DistributedBufferType, %arg2 : !OverlappedBufferType) -> !OverlappedBufferType {
    %alloc = memref.alloc() : memref<1x1792x288x4xf16, {order = #NHWC}, @DDR>
    %subview0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 896, 288, 4] : memref<1x1792x288x4xf16, {order = #NHWC}, @DDR> to memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>
    %copy0 = VPUIP.Copy inputs(%arg0 : !DistributedBufferType) outputs(%subview0 : memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) -> memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>

    %subview1 = VPUIP.SubView %alloc [0, 896, 0, 0] [1, 896, 288, 4] : memref<1x1792x288x4xf16, {order = #NHWC}, @DDR> to memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>
    %copy1 = VPUIP.Copy inputs(%arg1 : !DistributedBufferType) outputs(%subview1 : memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) -> memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>

    %concat = VPUIP.ConcatView inputs(%copy0, %copy1: memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>, memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) outputs(%alloc : memref<1x1792x288x4xf16, {order = #NHWC}, @DDR>) -> memref<1x1792x288x4xf16, {order = #NHWC}, @DDR>

    %subview2 = VPUIP.SubView %concat [0, 0, 0, 0] [1, 896, 288, 4] : memref<1x1792x288x4xf16, {order = #NHWC}, @DDR> to memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>
    %allocDistributed = VPURT.AllocDistributed -> !OverlappedBufferType

    %copy2 = VPUIP.Copy inputs(%subview2 : memref<1x896x288x4xf16, {order = #NHWC, strides = [2064384, 1, 7168, 1792]}, @DDR>) outputs(%allocDistributed : !OverlappedBufferType) -> !OverlappedBufferType

    %nceClusterTask = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 37893 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{eltwise_type = #VPU.eltwise_type<MULTIPLY>, is_inplace = true, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<ELTWISE>}>
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
    %10 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%9 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, {order = #NHWC}, @DDR>
    %11 = VPUIP.SubView %10 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>
    %12 = VPUIP.SubView %10 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>
    %13 = VPURT.AllocDistributed -> !ResultT
    %14 = VPUIP.Copy inputs(%11 : memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>) outputs(%13 : !ResultT) -> !ResultT
    %15 = VPURT.AllocDistributed -> !ResultT
    %16 = VPUIP.Copy inputs(%12 : memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>) outputs(%15 : !ResultT) -> !ResultT

    return %14, %16 : !ResultT, !ResultT

    // CHECK:                   [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[LEFT_INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[RIGHT_INPUT_ARG]] : memref<1x32x128x1xf32, @DDR>) -> memref<4096x1x1x1xf32, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf32, @DDR>) -> memref<4096x1x1x1xf32, {order = #NHWC}, @DDR>
    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, {order = #NHWC}, @DDR> to memref<128x1x1x1xf32, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_0]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_0:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_2]] : memref<128x1x1x1xf32, {order = #NHWC}, @DDR>)
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
    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, {order = #NHWC}, @DDR> to memref<128x1x1x1xf32, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_1:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_6]] : memref<128x1x1x1xf32, {order = #NHWC}, @DDR>)
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
    %9 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%8 : memref<4096x1024x1x1xf16, @DDR>) -> memref<4096x1024x1x1xf16, {order = #NHWC}, @DDR>
    %10 = VPUIP.SubView %9 [0, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>
    %11 = VPUIP.SubView %9 [128, 0, 0, 0] [128, 1024, 1, 1] : memref<4096x1024x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>
    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%10 : memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>) outputs(%12 : !ResultT) -> !ResultT
    %14 = VPURT.AllocDistributed -> !ResultT
    %15 = VPUIP.Copy inputs(%11 : memref<128x1024x1x1xf16, {order = #NHWC}, @DDR>) outputs(%14 : !ResultT) -> !ResultT

    return %13, %15 : !ResultT, !ResultT

    // CHECK:                   [[DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>
    // CHECK:                   [[GENERICRESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[INPUT_ARG]] : memref<1x32x128x1023xf16, @DDR>) -> memref<4096x1023x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_0]] : memref<4096x1023x1x1xf16, @DDR>) -> memref<4096x1023x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[ALLOC:%.+]] = memref.alloc() : memref<1x32x128x1xf32, @DDR>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1x32x128x1xf32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1], [1, 32, 32, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}>) outputs(
    // CHECK:                   [[ALLOC]] : memref<1x32x128x1xf32, @DDR>) -> memref<1x32x128x1xf32, @DDR>
    // CHECK:                   [[GENERICRESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x128x1xf32, @DDR>) -> memref<4096x1x1x1xf32, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[GENERICRESHAPE_1]] : memref<4096x1x1x1xf32, @DDR>) -> memref<4096x1x1x1xf32, {order = #NHWC}, @DDR>
    // CHECK:                   [[DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, {order = #NHWC}, @DDR> to memref<128x1x1x1xf32, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[DISTRIBUTED_1]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_0:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_2]] : memref<128x1x1x1xf32, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_3]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
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
    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [128, 0, 0, 0] [128, 1023, 1, 1] : memref<4096x1023x1x1xf16, {order = #NHWC}, @DDR> to memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 0, 0, 0] [128, 1023, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<128x1023x1x1xf16, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_5]] : !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<128x1023x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1], [32, 1023, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [128, 0, 0, 0] [128, 1, 1, 1] : memref<4096x1x1x1xf32, {order = #NHWC}, @DDR> to memref<128x1x1x1xf32, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[DISTRIBUTED_2]] [0, 1023, 0, 0] [128, 1, 1, 1] : !VPUIP.DistributedBuffer<128x1024x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1], [32, 1024, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1], [32, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [32, 0, 0, 0], [64, 0, 0, 0], [96, 0, 0, 0]]}>
    // CHECK:                   [[CONVERTDMA_1:%.+]] = VPUIP.ConvertDMA inputs([[SUBVIEW_6]] : memref<128x1x1x1xf32, {order = #NHWC}, @DDR>) outputs([[SUBVIEW_7]] : !VPUIP.DistributedBuffer<128x1x1x1xf16, {order = #NHWC, strides = [1024, 1, 1024, 1024]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
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
    %7 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs(%6 : memref<32768x96x1x1xf16, @DDR>) -> memref<32768x96x1x1xf16, {order = #NHWC}, @DDR>

    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1024, 96, 1, 1] : memref<32768x96x1x1xf16, {order = #NHWC}, @DDR> to memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>
    %9 = VPUIP.SubView %7 [1024, 0, 0, 0] [1024, 96, 1, 1] : memref<32768x96x1x1xf16, {order = #NHWC}, @DDR> to memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>

    %10 = VPURT.AllocDistributed -> !ResultT
    %11 = VPUIP.Copy inputs(%8 : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>) outputs(%10 : !ResultT) -> !ResultT

    %12 = VPURT.AllocDistributed -> !ResultT
    %13 = VPUIP.Copy inputs(%9 : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>) outputs(%12 : !ResultT) -> !ResultT

    return %11, %13 : !ResultT, !ResultT

    // CHECK:                   [[ALLOC:%.+]] = memref.alloc() : memref<1x32x256x96xf16, @DDR>
    // CHECK:                   [[COPY_0:%.+]] = VPUIP.Copy inputs([[ARG1]] : !VPUIP.DistributedBuffer<1x32x256x96xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96]], compute_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96], [1, 8, 256, 96]], memory_offsets = [[0, 0, 0, 0], [0, 8, 0, 0], [0, 16, 0, 0], [0, 24, 0, 0]]}>)
    // CHECK-SAME:                    outputs([[ALLOC:%.+]] : memref<1x32x256x96xf16, @DDR>) -> memref<1x32x256x96xf16, @DDR>

    // CHECK:                   [[RESHAPE_0:%.+]] = VPUIP.GenericReshape inputs([[ARG0]] : memref<1x32x768x96xf16, @DDR>) -> memref<24576x96x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE_0]] : memref<24576x96x1x1xf16, @DDR>) -> memref<24576x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[RESHAPE_1:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x32x256x96xf16, @DDR>) -> memref<8192x96x1x1xf16, @DDR>
    // CHECK:                   [[PERMUTECAST_1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RESHAPE_1]] : memref<8192x96x1x1xf16, @DDR>) -> memref<8192x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[ALLOC_DISTRIBUTED_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   [[ALLOC_0:%.+]] = memref.alloc() : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [0, 0, 0, 0] [768, 96, 1, 1] : memref<24576x96x1x1xf16, {order = #NHWC}, @DDR> to memref<768x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_1:%.+]] = VPUIP.SubView [[ALLOC_0:%.+]] [0, 0, 0, 0] [768, 96, 1, 1] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR> to memref<768x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[COPY_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]] : memref<768x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_1]] : memref<768x96x1x1xf16, {order = #NHWC}, @DDR>) -> memref<768x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[SUBVIEW_2:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [0, 0, 0, 0] [256, 96, 1, 1] : memref<8192x96x1x1xf16, {order = #NHWC}, @DDR> to memref<256x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_3:%.+]] = VPUIP.SubView [[ALLOC_0]] [768, 0, 0, 0] [256, 96, 1, 1] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR> to memref<256x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[COPY_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2]] : memref<256x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_3]] : memref<256x96x1x1xf16, {order = #NHWC}, @DDR>) -> memref<256x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[CONCATVIEW_0:%.+]] = VPUIP.ConcatView inputs([[COPY_1]], [[COPY_2]] : memref<768x96x1x1xf16, {order = #NHWC}, @DDR>, memref<256x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_0]] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[COPY_3:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_0]] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_DISTRIBUTED_0]] : !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   [[ALLOC_DISTRIBUTED_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1024x96x1x1xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:           compute_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], compute_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]],
    // CHECK-SAME{LITERAL}:           memory_shapes = [[256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1], [256, 96, 1, 1]], memory_offsets = [[0, 0, 0, 0], [256, 0, 0, 0], [512, 0, 0, 0], [768, 0, 0, 0]]}>

    // CHECK:                   [[ALLOC_1:%.+]] = memref.alloc() : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[SUBVIEW_4:%.+]] = VPUIP.SubView [[PERMUTECAST_0]] [768, 0, 0, 0] [768, 96, 1, 1] : memref<24576x96x1x1xf16, {order = #NHWC}, @DDR> to memref<768x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_5:%.+]] = VPUIP.SubView [[ALLOC_1:%.+]] [0, 0, 0, 0] [768, 96, 1, 1] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR> to memref<768x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[COPY_4:%.+]] = VPUIP.Copy inputs([[SUBVIEW_4]] : memref<768x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_5]] : memref<768x96x1x1xf16, {order = #NHWC}, @DDR>) -> memref<768x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[SUBVIEW_6:%.+]] = VPUIP.SubView [[PERMUTECAST_1]] [256, 0, 0, 0] [256, 96, 1, 1] : memref<8192x96x1x1xf16, {order = #NHWC}, @DDR> to memref<256x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[SUBVIEW_7:%.+]] = VPUIP.SubView [[ALLOC_1]] [768, 0, 0, 0] [256, 96, 1, 1] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR> to memref<256x96x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:                   [[COPY_5:%.+]] = VPUIP.Copy inputs([[SUBVIEW_6]] : memref<256x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[SUBVIEW_7]] : memref<256x96x1x1xf16, {order = #NHWC}, @DDR>) -> memref<256x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[CONCATVIEW_1:%.+]] = VPUIP.ConcatView inputs([[COPY_4]], [[COPY_5]] : memref<768x96x1x1xf16, {order = #NHWC}, @DDR>, memref<256x96x1x1xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:                    outputs([[ALLOC_1]] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>

    // CHECK:                   [[COPY_6:%.+]] = VPUIP.Copy inputs([[CONCATVIEW_1]] : memref<1024x96x1x1xf16, {order = #NHWC}, @DDR>)
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
    %alloc = memref.alloc() : memref<1x64x40x640xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>}, @DDR>
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
          -> memref<1x64x40x640xf16, {order = #NHWC}, @DDR>
    %11 = VPUIP.SubView %10 [0, 0, 0, 480] [1, 64, 40, 160]
          : memref<1x64x40x640xf16, {order = #NHWC}, @DDR>
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

// -----


// Concat shape  : 1x16384x4x2 NHWC (C=16384, H=4, W=2), 2 inputs each 1x8192x4x2
// perInputChannels = 16384/2 = 8192
// reshape chain : NHWC 1x16384x4x2 -> NHWC 1x16384x8x1 -> PermuteCast(d1,d3,d0,d2) ->
//                 NCHW 8x16384x1x1 -> flat 32x4096
// GatherDMA indices: [[0],[3]]
//   flat row  0 -> h=0,w=0,inputIdx=0,channelOffset=0    -> %arg0[0,    0,0,0][1,4096,1,1]
//   flat row  3 -> h=0,w=0,inputIdx=1,channelOffset=4096 -> %arg1[0, 4096,0,0][1,4096,1,1]
// Downstream ConcatView: 1x16x4096xf16, gather_a -> [0,0,0], gather_b→[0,8,0]

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

!UpstreamDist = !VPUIP.DistributedBuffer<
    1x8192x4x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 4096, 4, 2], [1, 4096, 4, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 4096, 0, 0]],
    memory_shapes  = [[1, 8192, 4, 2], [1, 8192, 4, 2]],
    memory_offsets  = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @FuseGatherDMAWithSubView
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
// CHECK-SAME: [[INPUT_2:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
// CHECK-SAME: [[INPUT_3:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
func.func @FuseGatherDMAWithSubView(
        %tile0a: !UpstreamDist, %tile1a: !UpstreamDist,
        %tile0b: !UpstreamDist, %tile1b: !UpstreamDist)
        -> memref<1x16x4096xf16, @DDR> {

    // Chain A
    %alloc_a = memref.alloc() : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>
    %sv_a0 = VPUIP.SubView %alloc_a [0,    0, 0, 0] [1, 8192, 4, 2] : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %cp_a0 = VPUIP.Copy inputs(%tile0a : !UpstreamDist) outputs(%sv_a0 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %sv_a1 = VPUIP.SubView %alloc_a [0, 8192, 0, 0] [1, 8192, 4, 2] : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %cp_a1 = VPUIP.Copy inputs(%tile1a : !UpstreamDist) outputs(%sv_a1 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %concat_a = VPUIP.ConcatView
        inputs(%cp_a0, %cp_a1 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>,
                                 memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>)
        outputs(%alloc_a : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>
    %rsh_a0 = VPUIP.GenericReshape inputs(%concat_a : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>) -> memref<1x16384x8x1xf16, {order = #NHWC}, @DDR>
    %perm_a  = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
               inputs(%rsh_a0 : memref<1x16384x8x1xf16, {order = #NHWC}, @DDR>) -> memref<8x16384x1x1xf16, @DDR>
    %flat_a  = VPUIP.GenericReshape inputs(%perm_a : memref<8x16384x1x1xf16, @DDR>) -> memref<32x4096xf16, @DDR>
    %cst_idx_a   = const.Declare memref<2x1xi64> = dense<[[0], [3]]> : tensor<2x1xi64>
    %idx_alloc_a = memref.alloc() : memref<2x1xi64, [@CMX_NN, 0]>
    %idx_copy_a  = VPUIP.Copy inputs(%cst_idx_a : memref<2x1xi64>) outputs(%idx_alloc_a : memref<2x1xi64, [@CMX_NN, 0]>) -> memref<2x1xi64, [@CMX_NN, 0]>
    %g_alloc_a   = memref.alloc() : memref<2x4096xf16, [@CMX_NN, 0]>
    %gather_a    = VPUIP.GatherDMA <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
                   inputs(%flat_a : memref<32x4096xf16, @DDR>)
                   indices(%idx_copy_a : memref<2x1xi64, [@CMX_NN, 0]>)
                   outputs(%g_alloc_a : memref<2x4096xf16, [@CMX_NN, 0]>)
                   -> memref<2x4096xf16, [@CMX_NN, 0]>
    %rsh_ga = VPUIP.GenericReshape inputs(%gather_a : memref<2x4096xf16, [@CMX_NN, 0]>) -> memref<1x2x4096xf16, [@CMX_NN, 0]>

    // Chain B
    %alloc_b = memref.alloc() : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>
    %sv_b0 = VPUIP.SubView %alloc_b [0,    0, 0, 0] [1, 8192, 4, 2] : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %cp_b0 = VPUIP.Copy inputs(%tile0b : !UpstreamDist) outputs(%sv_b0 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %sv_b1 = VPUIP.SubView %alloc_b [0, 8192, 0, 0] [1, 8192, 4, 2] : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %cp_b1 = VPUIP.Copy inputs(%tile1b : !UpstreamDist) outputs(%sv_b1 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>
    %concat_b = VPUIP.ConcatView
        inputs(%cp_b0, %cp_b1 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>,
                                 memref<1x8192x4x2xf16, {order = #NHWC, strides = [131072, 1, 32768, 16384]}, @DDR>)
        outputs(%alloc_b : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>
    %rsh_b0 = VPUIP.GenericReshape inputs(%concat_b : memref<1x16384x4x2xf16, {order = #NHWC}, @DDR>) -> memref<1x16384x8x1xf16, {order = #NHWC}, @DDR>
    %perm_b  = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
               inputs(%rsh_b0 : memref<1x16384x8x1xf16, {order = #NHWC}, @DDR>) -> memref<8x16384x1x1xf16, @DDR>
    %flat_b  = VPUIP.GenericReshape inputs(%perm_b : memref<8x16384x1x1xf16, @DDR>) -> memref<32x4096xf16, @DDR>
    %cst_idx_b   = const.Declare memref<2x1xi64> = dense<[[0], [3]]> : tensor<2x1xi64>
    %idx_alloc_b = memref.alloc() : memref<2x1xi64, [@CMX_NN, 0]>
    %idx_copy_b  = VPUIP.Copy inputs(%cst_idx_b : memref<2x1xi64>) outputs(%idx_alloc_b : memref<2x1xi64, [@CMX_NN, 0]>) -> memref<2x1xi64, [@CMX_NN, 0]>
    %g_alloc_b   = memref.alloc() : memref<2x4096xf16, [@CMX_NN, 0]>
    %gather_b    = VPUIP.GatherDMA <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
                   inputs(%flat_b : memref<32x4096xf16, @DDR>)
                   indices(%idx_copy_b : memref<2x1xi64, [@CMX_NN, 0]>)
                   outputs(%g_alloc_b : memref<2x4096xf16, [@CMX_NN, 0]>)
                   -> memref<2x4096xf16, [@CMX_NN, 0]>
    %rsh_gb = VPUIP.GenericReshape inputs(%gather_b : memref<2x4096xf16, [@CMX_NN, 0]>) -> memref<1x2x4096xf16, [@CMX_NN, 0]>

    // Downstream ConcatView
    %dst = memref.alloc() : memref<1x16x4096xf16, @DDR>
    %dst_sv_a = VPUIP.SubView %dst [0, 0, 0] [1, 2, 4096] : memref<1x16x4096xf16, @DDR>
                to memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    %copy_out_a = VPUIP.Copy
                inputs(%rsh_ga : memref<1x2x4096xf16, [@CMX_NN, 0]>)
                outputs(%dst_sv_a : memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>)
                -> memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    %dst_sv_b = VPUIP.SubView %dst [0, 8, 0] [1, 2, 4096] : memref<1x16x4096xf16, @DDR>
                to memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    %copy_out_b = VPUIP.Copy
                inputs(%rsh_gb : memref<1x2x4096xf16, [@CMX_NN, 0]>)
                outputs(%dst_sv_b : memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>)
                -> memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    %result = VPUIP.ConcatView
        inputs(%copy_out_a, %copy_out_b
            : memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>,
              memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>)
        outputs(%dst : memref<1x16x4096xf16, @DDR>)
        -> memref<1x16x4096xf16, @DDR>
    return %result : memref<1x16x4096xf16, @DDR>


    // CHECK-NOT: VPUIP.GatherDMA
    // CHECK:     [[DST:%.+]] = memref.alloc() : memref<1x16x4096xf16, @DDR>
    // gather_a slot [0,0,0][1,2,4096]
    // index 0 -> %arg0[0,0,0,0][1,4096,1,1]
    // CHECK:     [[DST_SV_A:%.+]] = VPUIP.SubView [[DST]] [0, 0, 0] [1, 2, 4096]
    // CHECK-SAME:    memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[DDR_SV_A0:%.+]] = VPUIP.SubView [[DST]] [0, 0, 0] [1, 1, 4096]
    // CHECK-SAME:    memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[CMX_SV_A0:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 0, 0] [1, 4096, 1, 1]
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4096x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN,
    // CHECK-SAME:    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64
    // CHECK:     [[RSH_A0:%.+]] = VPUIP.GenericReshape inputs([[DDR_SV_A0]]
    // CHECK-SAME:    -> memref<1x4096x1x1xf16, @DDR>
    // CHECK:     [[PERM_A0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RSH_A0]] : memref<1x4096x1x1xf16, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[COPY_A0:%.+]] = VPUIP.Copy inputs([[CMX_SV_A0]]
    // CHECK-SAME:    outputs([[PERM_A0]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[POST_RSH_A0:%.+]] = VPUIP.GenericReshape inputs([[COPY_A0]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>

    // index 3 -> %arg1[0,4096,0,0][1,4096,1,1]
    // CHECK:     [[DDR_SV_A1:%.+]] = VPUIP.SubView [[DST]] [0, 1, 0] [1, 1, 4096]
    // CHECK-SAME:    memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[CMX_SV_A1:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 4096, 0, 0] [1, 4096, 1, 1]
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4096x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN,
    // CHECK:     [[RSH_A1:%.+]] = VPUIP.GenericReshape inputs([[DDR_SV_A1]]
    // CHECK-SAME:    -> memref<1x4096x1x1xf16, @DDR>
    // CHECK:     [[PERM_A1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RSH_A1]] : memref<1x4096x1x1xf16, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[COPY_A1:%.+]] = VPUIP.Copy inputs([[CMX_SV_A1]]
    // CHECK-SAME:    outputs([[PERM_A1]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[POST_RSH_A1:%.+]] = VPUIP.GenericReshape inputs([[COPY_A1]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[REBUILT_A:%.+]] = VPUIP.ConcatView inputs([[POST_RSH_A0]], [[POST_RSH_A1]] : memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>, memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>) outputs([[DST_SV_A]] : memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>) -> memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>

    // gather_b slot [0,8,0][1,2,4096]
    // index 0 -> %arg2[0,0,0,0][1,4096,1,1]
    // CHECK:     [[DST_SV_B:%.+]] = VPUIP.SubView [[DST]] [0, 8, 0] [1, 2, 4096]
    // CHECK-SAME:    memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[DDR_SV_B0:%.+]] = VPUIP.SubView [[DST]] [0, 8, 0] [1, 1, 4096]
    // CHECK-SAME:    memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[CMX_SV_B0:%.+]] = VPUIP.SubView [[INPUT_2]] [0, 0, 0, 0] [1, 4096, 1, 1]
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4096x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN,
    // CHECK:     [[RSH_B0:%.+]] = VPUIP.GenericReshape inputs([[DDR_SV_B0]]
    // CHECK-SAME:    -> memref<1x4096x1x1xf16, @DDR>
    // CHECK:     [[PERM_B0:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RSH_B0]] : memref<1x4096x1x1xf16, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[COPY_B0:%.+]] = VPUIP.Copy inputs([[CMX_SV_B0]]
    // CHECK-SAME:    outputs([[PERM_B0]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[POST_RSH_B0:%.+]] = VPUIP.GenericReshape inputs([[COPY_B0]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>

    // index 3 -> %arg3[0,4096,0,0][1,4096,1,1]
    // CHECK:     [[DDR_SV_B1:%.+]] = VPUIP.SubView [[DST]] [0, 9, 0] [1, 1, 4096]
    // CHECK:     [[CMX_SV_B1:%.+]] = VPUIP.SubView [[INPUT_3]] [0, 4096, 0, 0] [1, 4096, 1, 1]
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x4096x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN,
    // CHECK:     [[RSH_B1:%.+]] = VPUIP.GenericReshape inputs([[DDR_SV_B1]]
    // CHECK-SAME:    -> memref<1x4096x1x1xf16, @DDR>
    // CHECK:     [[PERM_B1:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC} inputs([[RSH_B1]] : memref<1x4096x1x1xf16, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[COPY_B1:%.+]] = VPUIP.Copy inputs([[CMX_SV_B1]]
    // CHECK-SAME:    outputs([[PERM_B1]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:     [[POST_RSH_B1:%.+]] = VPUIP.GenericReshape inputs([[COPY_B1]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // CHECK:     [[REBUILT_B:%.+]] = VPUIP.ConcatView inputs([[POST_RSH_B0]], [[POST_RSH_B1]] : memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>, memref<1x1x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>) outputs([[DST_SV_B]] : memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>) -> memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>
    // Downstream ConcatView is preserved.
    // CHECK:     [[RESULT:%.+]] = VPUIP.ConcatView inputs([[REBUILT_A]], [[REBUILT_B]] : memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>, memref<1x2x4096xf16, {order = #CHW, strides = [65536, 4096, 1]}, @DDR>) outputs([[DST]] : memref<1x16x4096xf16, @DDR>) -> memref<1x16x4096xf16, @DDR>
    // CHECK:     return [[RESULT]] : memref<1x16x4096xf16, @DDR>
}

// -----

// FuseGatherDMAWithSubViewCopy negative case: no PermuteCast chain.

// CHECK-LABEL: func.func @NoFuseGatherWhenNoPermuteCast
func.func @NoFuseGatherWhenNoPermuteCast(%flat_ddr: memref<8x2xf16, @DDR>) -> memref<1x4x2xf16, @DDR> {
    %cst_indices  = const.Declare memref<2x1xi64> = dense<[[0], [3]]> : tensor<2x1xi64>
    %idx_alloc    = memref.alloc() : memref<2x1xi64, [@CMX_NN, 0]>
    %idx_copy     = VPUIP.Copy
        inputs(%cst_indices : memref<2x1xi64>)
        outputs(%idx_alloc  : memref<2x1xi64, [@CMX_NN, 0]>)
        -> memref<2x1xi64, [@CMX_NN, 0]>

    %gather_alloc = memref.alloc() : memref<2x2xf16, [@CMX_NN, 0]>
    %gather = VPUIP.GatherDMA
        <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
        inputs(%flat_ddr  : memref<8x2xf16, @DDR>)
        indices(%idx_copy : memref<2x1xi64, [@CMX_NN, 0]>)
        outputs(%gather_alloc : memref<2x2xf16, [@CMX_NN, 0]>)
        -> memref<2x2xf16, [@CMX_NN, 0]>

    %cmx_reshape = VPUIP.GenericReshape
        inputs(%gather : memref<2x2xf16, [@CMX_NN, 0]>)
        -> memref<1x2x2xf16, [@CMX_NN, 0]>

    %dst_alloc   = memref.alloc() : memref<1x4x2xf16, @DDR>
    %dst_subview = VPUIP.SubView %dst_alloc [0, 0, 0] [1, 2, 2]
        : memref<1x4x2xf16, @DDR>
        to memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>
    %final_copy = VPUIP.Copy
        inputs(%cmx_reshape : memref<1x2x2xf16, [@CMX_NN, 0]>)
        outputs(%dst_subview : memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>)
        -> memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>

    %concat_out = VPUIP.ConcatView
        inputs(%final_copy : memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>)
        outputs(%dst_alloc : memref<1x4x2xf16, @DDR>)
        -> memref<1x4x2xf16, @DDR>

    return %concat_out : memref<1x4x2xf16, @DDR>

    // CHECK: VPUIP.GatherDMA
}

// -----

// FuseGatherDMAWithSubViewCopy negative case: upstream concat inputs are SEGMENTED, not DUPLICATED.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!GatherSegBuf = !VPUIP.DistributedBuffer<
    1x2x2x2xf16, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 1, 2, 2], [1, 1, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes  = [[1, 1, 2, 2], [1, 1, 2, 2]],
    memory_offsets  = [[0, 0, 0, 0], [0, 1, 0, 0]]
}>

// CHECK-LABEL: func.func @NoFuseGatherWhenNotDuplicated
func.func @NoFuseGatherWhenNotDuplicated(%cmx0: !GatherSegBuf, %cmx1: !GatherSegBuf) -> memref<1x4x2xf16, @DDR> {
    %upstream_alloc = memref.alloc() : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>

    %subview_c0 = VPUIP.SubView %upstream_alloc [0, 0, 0, 0] [1, 2, 2, 2]
        : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>
        to memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>
    %copy_c0 = VPUIP.Copy
        inputs(%cmx0 : !GatherSegBuf)
        outputs(%subview_c0 : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>

    %subview_c1 = VPUIP.SubView %upstream_alloc [0, 2, 0, 0] [1, 2, 2, 2]
        : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>
        to memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>
    %copy_c1 = VPUIP.Copy
        inputs(%cmx1 : !GatherSegBuf)
        outputs(%subview_c1 : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>

    %upstream_concat = VPUIP.ConcatView
        inputs(%copy_c0, %copy_c1
            : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>,
              memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        outputs(%upstream_alloc : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x2x2xf16, {order = #NHWC}, @DDR>

    %reshape_before_permute = VPUIP.GenericReshape
        inputs(%upstream_concat : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x4x1xf16, {order = #NHWC}, @DDR>
    %permute = VPUIP.PermuteCast
        {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
        inputs(%reshape_before_permute : memref<1x4x4x1xf16, {order = #NHWC}, @DDR>)
        -> memref<4x4x1x1xf16, @DDR>
    %reshape_flat = VPUIP.GenericReshape
        inputs(%permute : memref<4x4x1x1xf16, @DDR>)
        -> memref<8x2xf16, @DDR>

    %cst_indices  = const.Declare memref<2x1xi64> = dense<[[0], [3]]> : tensor<2x1xi64>
    %idx_alloc    = memref.alloc() : memref<2x1xi64, [@CMX_NN, 0]>
    %idx_copy     = VPUIP.Copy
        inputs(%cst_indices : memref<2x1xi64>)
        outputs(%idx_alloc  : memref<2x1xi64, [@CMX_NN, 0]>)
        -> memref<2x1xi64, [@CMX_NN, 0]>

    %gather_alloc = memref.alloc() : memref<2x2xf16, [@CMX_NN, 0]>
    %gather = VPUIP.GatherDMA
        <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
        inputs(%reshape_flat : memref<8x2xf16, @DDR>)
        indices(%idx_copy    : memref<2x1xi64, [@CMX_NN, 0]>)
        outputs(%gather_alloc : memref<2x2xf16, [@CMX_NN, 0]>)
        -> memref<2x2xf16, [@CMX_NN, 0]>

    %cmx_reshape = VPUIP.GenericReshape
        inputs(%gather : memref<2x2xf16, [@CMX_NN, 0]>)
        -> memref<1x2x2xf16, [@CMX_NN, 0]>

    %dst_alloc   = memref.alloc() : memref<1x4x2xf16, @DDR>
    %dst_subview = VPUIP.SubView %dst_alloc [0, 0, 0] [1, 2, 2]
        : memref<1x4x2xf16, @DDR>
        to memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>
    %final_copy = VPUIP.Copy
        inputs(%cmx_reshape : memref<1x2x2xf16, [@CMX_NN, 0]>)
        outputs(%dst_subview : memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>)
        -> memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>

    %concat_out = VPUIP.ConcatView
        inputs(%final_copy : memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>)
        outputs(%dst_alloc : memref<1x4x2xf16, @DDR>)
        -> memref<1x4x2xf16, @DDR>

    return %concat_out : memref<1x4x2xf16, @DDR>

    // CHECK: VPUIP.GatherDMA
}

// -----

// FuseGatherDMAWithSubViewCopy negative case : GatherDMA indices come from a function argument.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!GatherDupBuf = !VPUIP.DistributedBuffer<
    1x2x2x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 2, 2, 2], [1, 2, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes  = [[1, 2, 2, 2], [1, 2, 2, 2]],
    memory_offsets  = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @NoFuseGatherWhenIndicesNotConst
func.func @NoFuseGatherWhenIndicesNotConst(
        %cmx0: !GatherDupBuf, %cmx1: !GatherDupBuf,
        %idx_arg: memref<2x1xi64, [@CMX_NN, 0]>) -> memref<1x4x2xf16, @DDR> {
    %upstream_alloc = memref.alloc() : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>

    %subview_c0 = VPUIP.SubView %upstream_alloc [0, 0, 0, 0] [1, 2, 2, 2]
        : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>
        to memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>
    %copy_c0 = VPUIP.Copy
        inputs(%cmx0 : !GatherDupBuf)
        outputs(%subview_c0 : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>

    %subview_c1 = VPUIP.SubView %upstream_alloc [0, 2, 0, 0] [1, 2, 2, 2]
        : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>
        to memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>
    %copy_c1 = VPUIP.Copy
        inputs(%cmx1 : !GatherDupBuf)
        outputs(%subview_c1 : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>

    %upstream_concat = VPUIP.ConcatView
        inputs(%copy_c0, %copy_c1
            : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>,
              memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        outputs(%upstream_alloc : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x2x2xf16, {order = #NHWC}, @DDR>

    %reshape_before_permute = VPUIP.GenericReshape
        inputs(%upstream_concat : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x4x1xf16, {order = #NHWC}, @DDR>
    %permute = VPUIP.PermuteCast
        {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
        inputs(%reshape_before_permute : memref<1x4x4x1xf16, {order = #NHWC}, @DDR>)
        -> memref<4x4x1x1xf16, @DDR>
    %reshape_flat = VPUIP.GenericReshape
        inputs(%permute : memref<4x4x1x1xf16, @DDR>)
        -> memref<8x2xf16, @DDR>

    %gather_alloc = memref.alloc() : memref<2x2xf16, [@CMX_NN, 0]>
    %gather = VPUIP.GatherDMA
        <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
        inputs(%reshape_flat : memref<8x2xf16, @DDR>)
        indices(%idx_arg     : memref<2x1xi64, [@CMX_NN, 0]>)
        outputs(%gather_alloc : memref<2x2xf16, [@CMX_NN, 0]>)
        -> memref<2x2xf16, [@CMX_NN, 0]>

    %cmx_reshape = VPUIP.GenericReshape
        inputs(%gather : memref<2x2xf16, [@CMX_NN, 0]>)
        -> memref<1x2x2xf16, [@CMX_NN, 0]>

    %dst_alloc   = memref.alloc() : memref<1x4x2xf16, @DDR>
    %dst_subview = VPUIP.SubView %dst_alloc [0, 0, 0] [1, 2, 2]
        : memref<1x4x2xf16, @DDR>
        to memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>
    %final_copy = VPUIP.Copy
        inputs(%cmx_reshape : memref<1x2x2xf16, [@CMX_NN, 0]>)
        outputs(%dst_subview : memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>)
        -> memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>

    %concat_out = VPUIP.ConcatView
        inputs(%final_copy : memref<1x2x2xf16, {order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, strides = [8, 2, 1]}, @DDR>)
        outputs(%dst_alloc : memref<1x4x2xf16, @DDR>)
        -> memref<1x4x2xf16, @DDR>

    return %concat_out : memref<1x4x2xf16, @DDR>

    // CHECK: VPUIP.GatherDMA
}


// -----

// FuseGatherDMAWithSubViewCopy negative: GatherDMA output is larger than its input.
// input 4x4xf16 (16 elements), output 8x4xf16 (32 elements) → skip optimization.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!GatherLargeBuf = !VPUIP.DistributedBuffer<
    1x2x2x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 2, 2, 2], [1, 2, 2, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes  = [[1, 2, 2, 2], [1, 2, 2, 2]],
    memory_offsets  = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @NoFuseGatherWhenOutputLargerThanInput
func.func @NoFuseGatherWhenOutputLargerThanInput(%cmx0: !GatherLargeBuf, %cmx1: !GatherLargeBuf) -> memref<1x8x4xf16, @DDR> {
    %upstream_alloc = memref.alloc() : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>

    %subview_c0 = VPUIP.SubView %upstream_alloc [0, 0, 0, 0] [1, 2, 2, 2]
        : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>
        to memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>
    %copy_c0 = VPUIP.Copy
        inputs(%cmx0 : !GatherLargeBuf)
        outputs(%subview_c0 : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>

    %subview_c1 = VPUIP.SubView %upstream_alloc [0, 2, 0, 0] [1, 2, 2, 2]
        : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>
        to memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>
    %copy_c1 = VPUIP.Copy
        inputs(%cmx1 : !GatherLargeBuf)
        outputs(%subview_c1 : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        -> memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>

    %upstream_concat = VPUIP.ConcatView
        inputs(%copy_c0, %copy_c1
            : memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>,
              memref<1x2x2x2xf16, {order = #NHWC, strides = [16, 1, 8, 4]}, @DDR>)
        outputs(%upstream_alloc : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x2x2xf16, {order = #NHWC}, @DDR>

    %reshape_before_permute = VPUIP.GenericReshape
        inputs(%upstream_concat : memref<1x4x2x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x4x4x1xf16, {order = #NHWC}, @DDR>
    %permute = VPUIP.PermuteCast
        {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
        inputs(%reshape_before_permute : memref<1x4x4x1xf16, {order = #NHWC}, @DDR>)
        -> memref<4x4x1x1xf16, @DDR>
    %reshape_flat = VPUIP.GenericReshape
        inputs(%permute : memref<4x4x1x1xf16, @DDR>)
        -> memref<4x4xf16, @DDR>

    // output (8x4 = 32 elements) > input (4x4 = 16 elements) → optimization must be skipped
    %cst_indices  = const.Declare memref<8x1xi64> = dense<[[0], [1], [2], [3], [0], [1], [2], [3]]> : tensor<8x1xi64>
    %idx_alloc    = memref.alloc() : memref<8x1xi64, [@CMX_NN, 0]>
    %idx_copy     = VPUIP.Copy
        inputs(%cst_indices : memref<8x1xi64>)
        outputs(%idx_alloc  : memref<8x1xi64, [@CMX_NN, 0]>)
        -> memref<8x1xi64, [@CMX_NN, 0]>

    %gather_alloc = memref.alloc() : memref<8x4xf16, [@CMX_NN, 0]>
    %gather = VPUIP.GatherDMA
        <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
        inputs(%reshape_flat : memref<4x4xf16, @DDR>)
        indices(%idx_copy    : memref<8x1xi64, [@CMX_NN, 0]>)
        outputs(%gather_alloc : memref<8x4xf16, [@CMX_NN, 0]>)
        -> memref<8x4xf16, [@CMX_NN, 0]>

    %cmx_reshape = VPUIP.GenericReshape
        inputs(%gather : memref<8x4xf16, [@CMX_NN, 0]>)
        -> memref<1x8x4xf16, [@CMX_NN, 0]>

    %dst_alloc = memref.alloc() : memref<1x8x4xf16, @DDR>
    %final_copy = VPUIP.Copy
        inputs(%cmx_reshape : memref<1x8x4xf16, [@CMX_NN, 0]>)
        outputs(%dst_alloc : memref<1x8x4xf16, @DDR>)
        -> memref<1x8x4xf16, @DDR>

    return %final_copy : memref<1x8x4xf16, @DDR>

    // CHECK: VPUIP.GatherDMA
}


// -----

//   numInputs=3, channelsPerInput=8192, gatherRowSize=12288, H=4, W=2
//   totalC = 3*8192 = 24576, rowsPerSpatialPosition = 24576/12288 = 2
//   numFlatRows = H*W*rowsPerSpatialPosition = 4*2*2 = 16
//   GatherDMA: inputs memref<16x12288xf16, @DDR>, indices memref<1x1xi64> = [[1]]
//
//   Row 1 indexing:
//     spatialPos = 1 / 2 = 0, channelBlock = 1 % 2 = 1
//     absOffset = 1 * 12288 = 12288, inputIdx = 12288 / 8192 = 1, chanOff = 12288 % 8192 = 4096
//     h = 0 / 2 = 0, w = 0 % 2 = 0
//     piece0: 8192-4096 = 4096 ch from in1 at chanOff=4096, h=0, w=0
//     piece1: 12288-4096 = 8192 ch from in2 at chanOff=0,    h=0, w=0
//
//   Each input: 1x8192x4x2xf16 NHWC DUPLICATED over 4 clusters.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

!LargeDupBuf = !VPUIP.DistributedBuffer<
    1x8192x4x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 4 : i64,
    uniform_distributed_segments,
    compute_shapes = [[1, 8192, 4, 2], [1, 8192, 4, 2], [1, 8192, 4, 2], [1, 8192, 4, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 8192, 4, 2], [1, 8192, 4, 2], [1, 8192, 4, 2], [1, 8192, 4, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @FuseGatherDMACrossDifferentInputs
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
// CHECK-SAME: [[INPUT_2:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x8192x4x2xf16
func.func @FuseGatherDMACrossDifferentInputs(
        %in0: !LargeDupBuf, %in1: !LargeDupBuf, %in2: !LargeDupBuf)
        -> memref<1x4x12288xf16, @DDR> {
    // Upstream concat: 3 inputs -> 1x24576x4x2xf16 NHWC DDR
    %alloc = memref.alloc() : memref<1x24576x4x2xf16, {order = #NHWC}, @DDR>
    %sv0  = VPUIP.SubView %alloc [0,     0, 0, 0] [1, 8192, 4, 2] : memref<1x24576x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>
    %cp0  = VPUIP.Copy inputs(%in0 : !LargeDupBuf) outputs(%sv0 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>
    %sv1  = VPUIP.SubView %alloc [0,  8192, 0, 0] [1, 8192, 4, 2] : memref<1x24576x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>
    %cp1  = VPUIP.Copy inputs(%in1 : !LargeDupBuf) outputs(%sv1 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>
    %sv2  = VPUIP.SubView %alloc [0, 16384, 0, 0] [1, 8192, 4, 2] : memref<1x24576x4x2xf16, {order = #NHWC}, @DDR> to memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>
    %cp2  = VPUIP.Copy inputs(%in2 : !LargeDupBuf) outputs(%sv2 : memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>) -> memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>
    %concat = VPUIP.ConcatView
        inputs(%cp0, %cp1, %cp2 :
            memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>,
            memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>,
            memref<1x8192x4x2xf16, {order = #NHWC, strides = [196608, 1, 49152, 24576]}, @DDR>)
        outputs(%alloc : memref<1x24576x4x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x24576x4x2xf16, {order = #NHWC}, @DDR>
    // reshapeBeforePermute: 1x24576x4x2 NHWC -> 1x24576x8x1 NHWC
    %rsh0 = VPUIP.GenericReshape inputs(%concat : memref<1x24576x4x2xf16, {order = #NHWC}, @DDR>) -> memref<1x24576x8x1xf16, {order = #NHWC}, @DDR>
    // PermuteCast (d1,d3,d0,d2): 1x24576x8x1 NHWC -> 8x24576x1x1 NCHW
    %perm = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
        inputs(%rsh0 : memref<1x24576x8x1xf16, {order = #NHWC}, @DDR>) -> memref<8x24576x1x1xf16, @DDR>
    // reshapeFlat: 8x24576x1x1 -> 16x12288 (numFlatRows=16, channelBlockSize=12288)
    %flat = VPUIP.GenericReshape inputs(%perm : memref<8x24576x1x1xf16, @DDR>) -> memref<16x12288xf16, @DDR>

    // GatherDMA: indices [[1],[0]], gather 2 rows from 16x12288 flat table
    %cst_idx = const.Declare memref<2x1xi64> = dense<[[1], [0]]> : tensor<2x1xi64>
    %idx_alloc = memref.alloc() : memref<2x1xi64, [@CMX_NN, 0]>
    %idx_copy = VPUIP.Copy inputs(%cst_idx : memref<2x1xi64>) outputs(%idx_alloc : memref<2x1xi64, [@CMX_NN, 0]>) -> memref<2x1xi64, [@CMX_NN, 0]>
    %g_alloc = memref.alloc() : memref<2x12288xf16, [@CMX_NN, 0]>
    %gather = VPUIP.GatherDMA <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
        inputs(%flat : memref<16x12288xf16, @DDR>)
        indices(%idx_copy : memref<2x1xi64, [@CMX_NN, 0]>)
        outputs(%g_alloc : memref<2x12288xf16, [@CMX_NN, 0]>)
        -> memref<2x12288xf16, [@CMX_NN, 0]>
    // reshape gather output: 2x12288 -> 1x2x12288
    %rsh_g = VPUIP.GenericReshape inputs(%gather : memref<2x12288xf16, [@CMX_NN, 0]>) -> memref<1x2x12288xf16, [@CMX_NN, 0]>

    // Downstream: copy whole 1x2x12288 block into DDR SubView slot, assembled by ConcatView
    %dst = memref.alloc() : memref<1x4x12288xf16, @DDR>
    %dst_sv = VPUIP.SubView %dst [0, 0, 0] [1, 2, 12288]
        : memref<1x4x12288xf16, @DDR>
        to memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    %copy_out = VPUIP.Copy
        inputs(%rsh_g : memref<1x2x12288xf16, [@CMX_NN, 0]>)
        outputs(%dst_sv : memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>)
        -> memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    %dummy_cmx = memref.alloc() : memref<1x2x12288xf16, [@CMX_NN, 0]>
    %dst_sv2 = VPUIP.SubView %dst [0, 2, 0] [1, 2, 12288]
        : memref<1x4x12288xf16, @DDR>
        to memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    %copy_out2 = VPUIP.Copy
        inputs(%dummy_cmx : memref<1x2x12288xf16, [@CMX_NN, 0]>)
        outputs(%dst_sv2 : memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>)
        -> memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    %result = VPUIP.ConcatView
        inputs(%copy_out, %copy_out2
            : memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>,
              memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>)
        outputs(%dst : memref<1x4x12288xf16, @DDR>)
        -> memref<1x4x12288xf16, @DDR>
    return %result : memref<1x4x12288xf16, @DDR>

    // GatherDMA is fused away. 2 rows gathered:
    // Row 0 (index=1): spans in1 and in2:
    //   piece0: 4096 ch from %arg1 at chanOff=4096, h=0, w=0
    //   piece1: 8192 ch from %arg2 at chanOff=0,    h=0, w=0
    // Row 1 (index=0): spans in0 and in1:
    //   piece0: 8192 ch from %arg0 at chanOff=0, h=0, w=0
    //   piece1: 4096 ch from %arg1 at chanOff=0, h=0, w=0
    //
    // CHECK-NOT: VPUIP.GatherDMA
    //
    // CHECK:       [[OUT_ALLOC:%.+]] = memref.alloc() : memref<1x4x12288xf16, @DDR>
    // CHECK:       [[GATHERED_SV:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0] [1, 2, 12288]
    // CHECK-SAME:      memref<1x4x12288xf16, @DDR> to memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // --- row 0 (index=1) ---
    // CHECK:       [[ROW0_SV:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 0, 0] [1, 1, 12288]
    // CHECK-SAME:      memref<1x4x12288xf16, @DDR> to memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R0P0_CMX:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 4096, 0, 0] [1, 4096, 1, 1]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x8192x4x2xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x4096x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN
    // CHECK:       [[R0P0_DDR:%.+]] = VPUIP.SubView [[ROW0_SV]] [0, 0, 0] [1, 1, 4096]
    // CHECK-SAME:      memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR> to memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R0P0_RSH:%.+]] = VPUIP.GenericReshape inputs([[R0P0_DDR]]
    // CHECK-SAME:      memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x4096x1x1xf16, @DDR>
    // CHECK:       [[R0P0_PERM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[R0P0_RSH]] : memref<1x4096x1x1xf16, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R0P0_COPY:%.+]] = VPUIP.Copy inputs([[R0P0_CMX]]
    // CHECK-SAME:      outputs([[R0P0_PERM]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R0P0_OUT:%.+]] = VPUIP.GenericReshape inputs([[R0P0_COPY]]
    // CHECK-SAME:      memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R0P1_CMX:%.+]] = VPUIP.SubView [[INPUT_2]] [0, 0, 0, 0] [1, 8192, 1, 1]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x8192x4x2xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x8192x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN
    // CHECK:       [[R0P1_DDR:%.+]] = VPUIP.SubView [[ROW0_SV]] [0, 0, 4096] [1, 1, 8192]
    // CHECK-SAME:      memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR> to memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R0P1_RSH:%.+]] = VPUIP.GenericReshape inputs([[R0P1_DDR]]
    // CHECK-SAME:      memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x8192x1x1xf16, @DDR>
    // CHECK:       [[R0P1_PERM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[R0P1_RSH]] : memref<1x8192x1x1xf16, @DDR>) -> memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R0P1_COPY:%.+]] = VPUIP.Copy inputs([[R0P1_CMX]]
    // CHECK-SAME:      outputs([[R0P1_PERM]] : memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R0P1_OUT:%.+]] = VPUIP.GenericReshape inputs([[R0P1_COPY]]
    // CHECK-SAME:      memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[ROW0:%.+]] = VPUIP.ConcatView inputs([[R0P0_OUT]], [[R0P1_OUT]] : memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>, memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) outputs([[ROW0_SV]] : memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // --- row 1 (index=0) ---
    // CHECK:       [[ROW1_SV:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 1, 0] [1, 1, 12288]
    // CHECK-SAME:      memref<1x4x12288xf16, @DDR> to memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R1P0_CMX:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 0, 0, 0] [1, 8192, 1, 1]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x8192x4x2xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x8192x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN
    // CHECK:       [[R1P0_DDR:%.+]] = VPUIP.SubView [[ROW1_SV]] [0, 0, 0] [1, 1, 8192]
    // CHECK-SAME:      memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR> to memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R1P0_RSH:%.+]] = VPUIP.GenericReshape inputs([[R1P0_DDR]]
    // CHECK-SAME:      memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x8192x1x1xf16, @DDR>
    // CHECK:       [[R1P0_PERM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[R1P0_RSH]] : memref<1x8192x1x1xf16, @DDR>) -> memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R1P0_COPY:%.+]] = VPUIP.Copy inputs([[R1P0_CMX]]
    // CHECK-SAME:      outputs([[R1P0_PERM]] : memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R1P0_OUT:%.+]] = VPUIP.GenericReshape inputs([[R1P0_COPY]]
    // CHECK-SAME:      memref<1x8192x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R1P1_CMX:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 4096, 1, 1]
    // CHECK-SAME:      !VPUIP.DistributedBuffer<1x8192x4x2xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      to !VPUIP.DistributedBuffer<1x4096x1x1xf16, {order = #NHWC, strides = [65536, 1, 16384, 8192]}, @CMX_NN
    // CHECK:       [[R1P1_DDR:%.+]] = VPUIP.SubView [[ROW1_SV]] [0, 0, 8192] [1, 1, 4096]
    // CHECK-SAME:      memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR> to memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[R1P1_RSH:%.+]] = VPUIP.GenericReshape inputs([[R1P1_DDR]]
    // CHECK-SAME:      memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x4096x1x1xf16, @DDR>
    // CHECK:       [[R1P1_PERM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[R1P1_RSH]] : memref<1x4096x1x1xf16, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R1P1_COPY:%.+]] = VPUIP.Copy inputs([[R1P1_CMX]]
    // CHECK-SAME:      outputs([[R1P1_PERM]] : memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[R1P1_OUT:%.+]] = VPUIP.GenericReshape inputs([[R1P1_COPY]]
    // CHECK-SAME:      memref<1x4096x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[ROW1:%.+]] = VPUIP.ConcatView inputs([[R1P0_OUT]], [[R1P1_OUT]] : memref<1x1x8192xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>, memref<1x1x4096xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) outputs([[ROW1_SV]] : memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[GATHERED:%.+]] = VPUIP.ConcatView inputs([[ROW0]], [[ROW1]] : memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>, memref<1x1x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) outputs([[GATHERED_SV]] : memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[DUMMY_CMX:%.+]] = memref.alloc() : memref<1x2x12288xf16, [@CMX_NN, 0]>
    // CHECK:       [[SLOT1_SV:%.+]] = VPUIP.SubView [[OUT_ALLOC]] [0, 2, 0] [1, 2, 12288]
    // CHECK-SAME:      memref<1x4x12288xf16, @DDR> to memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[SLOT1_COPY:%.+]] = VPUIP.Copy inputs([[DUMMY_CMX]] : memref<1x2x12288xf16, [@CMX_NN, 0]>) outputs([[SLOT1_SV]] : memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) -> memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>
    // CHECK:       [[RET:%.+]] = VPUIP.ConcatView inputs([[GATHERED]], [[SLOT1_COPY]] : memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>, memref<1x2x12288xf16, {order = #CHW, strides = [49152, 12288, 1]}, @DDR>) outputs([[OUT_ALLOC]] : memref<1x4x12288xf16, @DDR>) -> memref<1x4x12288xf16, @DDR>
    // CHECK:       return [[RET]] : memref<1x4x12288xf16, @DDR>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

!UpstreamDist7984 = !VPUIP.DistributedBuffer<
    1x7984x4x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 3992, 4, 2], [1, 3992, 4, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 3992, 0, 0]],
    memory_shapes = [[1, 7984, 4, 2], [1, 7984, 4, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!UpstreamDist7840 = !VPUIP.DistributedBuffer<
    1x7840x4x2xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED",
    num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1],
    uniform_distributed_segments,
    compute_shapes = [[1, 3920, 4, 2], [1, 3920, 4, 2]],
    compute_offsets = [[0, 0, 0, 0], [0, 3920, 0, 0]],
    memory_shapes = [[1, 7840, 4, 2], [1, 7840, 4, 2]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: func.func @FuseGatherDMAWithUnbalancedInputChannels

// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x7984x4x2xf16
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: !VPUIP.DistributedBuffer<1x7840x4x2xf16
func.func @FuseGatherDMAWithUnbalancedInputChannels(%in0: !UpstreamDist7984, %in1: !UpstreamDist7840)
        -> memref<1x2x688xf16, @DDR> {
    %alloc = memref.alloc() : memref<1x15824x4x2xf16, {order = #NHWC}, @DDR>
    %sv0 = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 7984, 4, 2]
        : memref<1x15824x4x2xf16, {order = #NHWC}, @DDR>
        to memref<1x7984x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>
    %cp0 = VPUIP.Copy inputs(%in0 : !UpstreamDist7984)
        outputs(%sv0 : memref<1x7984x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>)
        -> memref<1x7984x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>
    %sv1 = VPUIP.SubView %alloc [0, 7984, 0, 0] [1, 7840, 4, 2]
        : memref<1x15824x4x2xf16, {order = #NHWC}, @DDR>
        to memref<1x7840x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>
    %cp1 = VPUIP.Copy inputs(%in1 : !UpstreamDist7840)
        outputs(%sv1 : memref<1x7840x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>)
        -> memref<1x7840x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>

    %concat = VPUIP.ConcatView
        inputs(%cp0, %cp1 : memref<1x7984x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>,
                           memref<1x7840x4x2xf16, {order = #NHWC, strides = [126592, 1, 31648, 15824]}, @DDR>)
        outputs(%alloc : memref<1x15824x4x2xf16, {order = #NHWC}, @DDR>)
        -> memref<1x15824x4x2xf16, {order = #NHWC}, @DDR>
    %rsh0 = VPUIP.GenericReshape inputs(%concat : memref<1x15824x4x2xf16, {order = #NHWC}, @DDR>) -> memref<1x15824x8x1xf16, {order = #NHWC}, @DDR>
    %perm = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>}
        inputs(%rsh0 : memref<1x15824x8x1xf16, {order = #NHWC}, @DDR>) -> memref<8x15824x1x1xf16, @DDR>
    %flat = VPUIP.GenericReshape inputs(%perm : memref<8x15824x1x1xf16, @DDR>) -> memref<184x688xf16, @DDR>

    %cst_idx = const.Declare memref<1x1xi64> = dense<[[11]]> : tensor<1x1xi64>
    %idx_alloc = memref.alloc() : memref<1x1xi64, [@CMX_NN, 0]>
    %idx_copy = VPUIP.Copy inputs(%cst_idx : memref<1x1xi64>) outputs(%idx_alloc : memref<1x1xi64, [@CMX_NN, 0]>) -> memref<1x1xi64, [@CMX_NN, 0]>
    %g_alloc = memref.alloc() : memref<1x688xf16, [@CMX_NN, 0]>
    %gather = VPUIP.GatherDMA <{addressingMode = 1 : i64, elementSize = 0 : i64, padding = 0 : i64, port = 0 : i64}>
        inputs(%flat : memref<184x688xf16, @DDR>)
        indices(%idx_copy : memref<1x1xi64, [@CMX_NN, 0]>)
        outputs(%g_alloc : memref<1x688xf16, [@CMX_NN, 0]>)
        -> memref<1x688xf16, [@CMX_NN, 0]>
    %rsh_g = VPUIP.GenericReshape inputs(%gather : memref<1x688xf16, [@CMX_NN, 0]>) -> memref<1x1x688xf16, [@CMX_NN, 0]>

    %dst = memref.alloc() : memref<1x2x688xf16, @DDR>
    %dst_sv = VPUIP.SubView %dst [0, 0, 0] [1, 1, 688]
        : memref<1x2x688xf16, @DDR>
        to memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    %copy_out = VPUIP.Copy
        inputs(%rsh_g : memref<1x1x688xf16, [@CMX_NN, 0]>)
        outputs(%dst_sv : memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>)
        -> memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>

    %dummy_cmx = memref.alloc() : memref<1x1x688xf16, [@CMX_NN, 0]>
    %dst_sv2 = VPUIP.SubView %dst [0, 1, 0] [1, 1, 688]
        : memref<1x2x688xf16, @DDR>
        to memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    %copy_out2 = VPUIP.Copy
        inputs(%dummy_cmx : memref<1x1x688xf16, [@CMX_NN, 0]>)
        outputs(%dst_sv2 : memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>)
        -> memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>

    %result = VPUIP.ConcatView
        inputs(%copy_out, %copy_out2 : memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>,
                                       memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>)
        outputs(%dst : memref<1x2x688xf16, @DDR>)
        -> memref<1x2x688xf16, @DDR>
    return %result : memref<1x2x688xf16, @DDR>

    // CHECK-NOT: VPUIP.GatherDMA
    // CHECK: [[DST:%.+]] = memref.alloc() : memref<1x2x688xf16, @DDR>
    // CHECK: [[SLOT0:%.+]] = VPUIP.SubView [[DST]] [0, 0, 0] [1, 1, 688]
    // CHECK-SAME: memref<1x2x688xf16, @DDR> to memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[R0P0_CMX:%.+]] = VPUIP.SubView [[INPUT_0]] [0, 7568, 0, 0] [1, 416, 1, 1]
    // CHECK-SAME: !VPUIP.DistributedBuffer<1x7984x4x2xf16
    // CHECK: [[R0P0_DDR:%.+]] = VPUIP.SubView [[SLOT0]] [0, 0, 0] [1, 1, 416]
    // CHECK-SAME: memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR> to memref<1x1x416xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[R0P0_RSH:%.+]] = VPUIP.GenericReshape inputs([[R0P0_DDR]]
    // CHECK-SAME: : memref<1x1x416xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>) -> memref<1x416x1x1xf16, @DDR>
    // CHECK: [[R0P0_PERM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME: inputs([[R0P0_RSH]] : memref<1x416x1x1xf16, @DDR>) -> memref<1x416x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK: [[R0P0_COPY:%.+]] = VPUIP.Copy inputs([[R0P0_CMX]]
    // CHECK-SAME: outputs([[R0P0_PERM]] : memref<1x416x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x416x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK: [[R0P0_OUT:%.+]] = VPUIP.GenericReshape inputs([[R0P0_COPY]]
    // CHECK-SAME: ) -> memref<1x1x416xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[R0P1_CMX:%.+]] = VPUIP.SubView [[INPUT_1]] [0, 0, 0, 0] [1, 272, 1, 1]
    // CHECK-SAME: !VPUIP.DistributedBuffer<1x7840x4x2xf16
    // CHECK: [[R0P1_DDR:%.+]] = VPUIP.SubView [[SLOT0]] [0, 0, 416] [1, 1, 272]
    // CHECK-SAME: memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR> to memref<1x1x272xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[R0P1_RSH:%.+]] = VPUIP.GenericReshape inputs([[R0P1_DDR]]
    // CHECK-SAME: : memref<1x1x272xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>) -> memref<1x272x1x1xf16, @DDR>
    // CHECK: [[R0P1_PERM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME: inputs([[R0P1_RSH]] : memref<1x272x1x1xf16, @DDR>) -> memref<1x272x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK: [[R0P1_COPY:%.+]] = VPUIP.Copy inputs([[R0P1_CMX]]
    // CHECK-SAME: outputs([[R0P1_PERM]] : memref<1x272x1x1xf16, {order = #NHWC}, @DDR>) -> memref<1x272x1x1xf16, {order = #NHWC}, @DDR>
    // CHECK: [[R0P1_OUT:%.+]] = VPUIP.GenericReshape inputs([[R0P1_COPY]]
    // CHECK-SAME: ) -> memref<1x1x272xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[ROW0_REBUILT:%.+]] = VPUIP.ConcatView inputs([[R0P0_OUT]], [[R0P1_OUT]]
    // CHECK-SAME: outputs([[SLOT0]] : memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>) -> memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[DUMMY:%.+]] = memref.alloc() : memref<1x1x688xf16, [@CMX_NN, 0]>
    // CHECK: [[SLOT1:%.+]] = VPUIP.SubView [[DST]] [0, 1, 0] [1, 1, 688]
    // CHECK-SAME: memref<1x2x688xf16, @DDR> to memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[COPY1:%.+]] = VPUIP.Copy inputs([[DUMMY]] : memref<1x1x688xf16, [@CMX_NN, 0]>) outputs([[SLOT1]] : memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>) -> memref<1x1x688xf16, {order = #CHW, strides = [1376, 688, 1]}, @DDR>
    // CHECK: [[RET:%.+]] = VPUIP.ConcatView inputs([[ROW0_REBUILT]], [[COPY1]]
    // CHECK-SAME: outputs([[DST]] : memref<1x2x688xf16, @DDR>) -> memref<1x2x688xf16, @DDR>
    // CHECK: return [[RET]] : memref<1x2x688xf16, @DDR>
}
