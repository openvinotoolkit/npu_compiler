//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%"  --convert-func-args-to-declarations --canonicalize --move-declarations-to-top %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DDDDDD attributes {VPU.debatch = 1 : i64} {
    VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
    module @VPU.SW {
        func.func private @builtin_SoftMax(memref<*xf16>, memref<*xf16>, i64, i64) attributes {VPU.kernel_code = "softmax.cpp", VPU.kernel_entry = "softmax", VPU.task_type = @COMPUTE}
        func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
    }

    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<3x3x224x224xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<3x1000xf16>
    }

    // CHECK: func.func private @batching([[ARG0:%.+]]: memref<1x3x224x224xf16, @DDR>, [[ARG1:%.+]]: memref<1x1000xf16, @DDR>)
    func.func private @batching(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR> {
        %3 = VPURT.DeclareBuffer <CMX_NN> <512> -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 3, 38, 224], [1, 3, 38, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224]], compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 113, 0], [0, 0, 150, 0], [0, 0, 187, 0]], memory_shapes = [[1, 3, 38, 224], [1, 3, 38, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224]], memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 113, 0], [0, 0, 150, 0], [0, 0, 187, 0]]}>
        %6 = VPURT.DeclareBuffer <CMX_NN> <122368> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %189 = VPURT.DeclareBuffer <CMX_NN> <761856> -> !VPUIP.DistributedBuffer<1x1008x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 688, 0, 0], [0, 848, 0, 0]], memory_shapes = [[1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %190 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%190 : !VPURT.Barrier) {
            %650 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x3x224x224xf16, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 3, 38, 224], [1, 3, 38, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224]], compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 113, 0], [0, 0, 150, 0], [0, 0, 187, 0]], memory_shapes = [[1, 3, 38, 224], [1, 3, 38, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224]], memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 113, 0], [0, 0, 150, 0], [0, 0, 187, 0]]}>) -> !VPUIP.DistributedBuffer<1x3x224x224xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 6, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 3, 38, 224], [1, 3, 38, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224]], compute_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 113, 0], [0, 0, 150, 0], [0, 0, 187, 0]], memory_shapes = [[1, 3, 38, 224], [1, 3, 38, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224], [1, 3, 37, 224]], memory_offsets = [[0, 0, 0, 0], [0, 0, 38, 0], [0, 0, 76, 0], [0, 0, 113, 0], [0, 0, 150, 0], [0, 0, 187, 0]]}>
        }
        %647 = VPUIP.GenericReshape inputs(%arg1 : memref<1x1000xf16, @DDR>) -> memref<1x1000x1x1xf16, @DDR>
        %648 = VPUIP.SubView %189 [0, 0, 0, 0] [1, 1000, 1, 1] : !VPUIP.DistributedBuffer<1x1008x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 176, 1, 1], [1, 176, 1, 1], [1, 176, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1], [1, 160, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 176, 0, 0], [0, 352, 0, 0], [0, 528, 0, 0], [0, 688, 0, 0], [0, 848, 0, 0]], memory_shapes = [[1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1], [1, 1008, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> to !VPUIP.DistributedBuffer<1x1000x1x1xf16, {order = #NHWC, strides = [1008, 1, 1008, 1008]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
        %649 = VPUIP.PermuteCast {dst_order = #NCHW, mem_perm = #NWCH} inputs(%647 : memref<1x1000x1x1xf16, @DDR>) -> memref<1x1000x1x1xf16, #NHWC, @DDR>
        VPURT.Task waits(%190 : !VPURT.Barrier) {
            %650 = VPUIP.NNDMA {port = 0 : i64} inputs(%648 : !VPUIP.DistributedBuffer<1x1000x1x1xf16, {order = #NHWC, strides = [1008, 1, 1008, 1008]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 6, 1, 1], num_clusters = 6 : i64, uniform_distributed_segments, compute_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1], [1, 1000, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) outputs(%649 : memref<1x1000x1x1xf16, #NHWC, @DDR>) -> memref<1x1000x1x1xf16, #NHWC, @DDR>
        }
        return %arg1 : memref<1x1000xf16, @DDR>
    }
    //CHECK: [[ARG_0:%.+]] = VPURT.DeclareBuffer <FunctionInput> [0] <0> -> memref<1x3x224x224xf16, @DDR>
    //CHECK: [[ARG_1:%.+]] = VPURT.DeclareBuffer <FunctionInput> [1] <903168> -> memref<1x1000xf16, @DDR>
    //CHECK: [[CMX_NON_MOD:%.+]] = VPURT.DeclareBuffer <CMX_NN> <512> -> !VPUIP.DistributedBuffer<1x3x224x224xf16, [[NOT_IMPORTANT_MATCH:.*]]

    // CHECK: func.func private @outline1([[ARG0:%.+]]: memref<3x3x224x224xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x224x224xf16, @DDR>)
    func.func private @outline1(%arg0: memref<3x3x224x224xf16, @DDR>, %arg1: memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 3, 224, 224] : memref<3x3x224x224xf16, @DDR> to memref<1x3x224x224xf16, @DDR>
        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x3x224x224xf16, @DDR>) outputs(%arg1 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        }
        return %arg1 : memref<1x3x224x224xf16, @DDR>
    }
    //CHECK: [[ARG_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<3x3x224x224xf16, @DDR>
    //CHECK: [[ARG_1:%.+]] = VPURT.DeclareBuffer <FunctionInput> [1] <0> -> memref<1x3x224x224xf16, @DDR>
    //CHECK: [[SUBVIEW_NON_MOD:%.+]] = VPUIP.SubView [[ARG_0]] [0, 0, 0, 0] [1, 3, 224, 224] [[NOT_IMPORTANT_MATCH:.*]]


    // CHECK: func.func private @outline2([[ARG0:%.+]]: memref<3x3x224x224xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x224x224xf16, @DDR>)
    func.func private @outline2(%arg0: memref<3x3x224x224xf16, @DDR>, %arg1: memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPUIP.SubView %arg0 [1, 0, 0, 0] [1, 3, 224, 224] : memref<3x3x224x224xf16, @DDR> to memref<1x3x224x224xf16, @DDR>
        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x3x224x224xf16, @DDR>) outputs(%arg1 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        }
        return %arg1 : memref<1x3x224x224xf16, @DDR>
    }
    //CHECK: [[ARG_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<3x3x224x224xf16, @DDR>
    //CHECK: [[ARG_1:%.+]] = VPURT.DeclareBuffer <FunctionInput> [1] <301056> -> memref<1x3x224x224xf16, @DDR>
    //CHECK: [[SUBVIEW_NON_MOD:%.+]] = VPUIP.SubView [[ARG_0]] [1, 0, 0, 0] [1, 3, 224, 224] [[NOT_IMPORTANT_MATCH:.*]]


    // CHECK: func.func private @outline3([[ARG0:%.+]]: memref<3x3x224x224xf16, @DDR>, [[ARG1:%.+]]: memref<1x3x224x224xf16, @DDR>)
    func.func private @outline3(%arg0: memref<3x3x224x224xf16, @DDR>, %arg1: memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPUIP.SubView %arg0 [2, 0, 0, 0] [1, 3, 224, 224] : memref<3x3x224x224xf16, @DDR> to memref<1x3x224x224xf16, @DDR>
        VPURT.Task {
            %2 = VPUIP.NNDMA {port = 0 : i64} inputs(%1 : memref<1x3x224x224xf16, @DDR>) outputs(%arg1 : memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        }
        return %arg1 : memref<1x3x224x224xf16, @DDR>
    }
    //CHECK: [[ARG_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<3x3x224x224xf16, @DDR>
    //CHECK: [[ARG_1:%.+]] = VPURT.DeclareBuffer <FunctionInput> [1] <602112> -> memref<1x3x224x224xf16, @DDR>
    //CHECK: [[SUBVIEW_NON_MOD:%.+]] = VPUIP.SubView [[ARG_0]] [2, 0, 0, 0] [1, 3, 224, 224] [[NOT_IMPORTANT_MATCH:.*]]


    // CHECK: func.func private @outline4([[ARG0:%.+]]: memref<1x1000xf16, @DDR>, [[ARG1:%.+]]: memref<1x1000xf16, @DDR>, [[ARG2:%.+]]: memref<1x1000xf16, @DDR>, [[ARG3:%.+]]: memref<3x1000xf16, @DDR>)
    func.func private @outline4(%arg0: memref<1x1000xf16, @DDR>, %arg1: memref<1x1000xf16, @DDR>, %arg2: memref<1x1000xf16, @DDR>, %arg3: memref<3x1000xf16, @DDR>) -> memref<3x1000xf16, @DDR> {
        %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %1 = VPUIP.SubView %arg3 [0, 0] [1, 1000] : memref<3x1000xf16, @DDR> to memref<1x1000xf16, @DDR>
        VPURT.Task {
            %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1000xf16, @DDR>) outputs(%1 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        }
        %2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %3 = VPUIP.SubView %arg3 [1, 0] [1, 1000] : memref<3x1000xf16, @DDR> to memref<1x1000xf16, @DDR>
        VPURT.Task {
            %6 = VPUIP.NNDMA {port = 1 : i64} inputs(%arg1 : memref<1x1000xf16, @DDR>) outputs(%3 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        }
        %4 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %5 = VPUIP.SubView %arg3 [2, 0] [1, 1000] : memref<3x1000xf16, @DDR> to memref<1x1000xf16, @DDR>
        VPURT.Task {
            %6 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg2 : memref<1x1000xf16, @DDR>) outputs(%5 : memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        }
        return %arg3 : memref<3x1000xf16, @DDR>
    }

    //CHECK: [[ARG_0:%.+]] = VPURT.DeclareBuffer <FunctionInput> [0] <903168> -> memref<1x1000xf16, @DDR>
    //CHECK: [[ARG_1:%.+]] = VPURT.DeclareBuffer <FunctionInput> [1] <0> -> memref<1x1000xf16, @DDR>
    //CHECK: [[ARG_2:%.+]] = VPURT.DeclareBuffer <FunctionInput> [2] <2048> -> memref<1x1000xf16, @DDR>
    //CHECK: [[ARG_3:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<3x1000xf16, @DDR>
    //CHECK: [[SUBVIEW_NON_MOD:%.+]] = VPUIP.SubView [[ARG_3]] [0, 0] [1, 1000] [[NOT_IMPORTANT_MATCH:.*]]

    func.func @main(%arg0: memref<3x3x224x224xf16, @DDR>, %arg1: memref<3x1000xf16, @DDR>) -> memref<3x1000xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224xf16, @DDR>
        %1 = VPURT.DeclareBuffer <DDR> <903168> -> memref<1x1000xf16, @DDR>
        %2 = VPURT.DeclareBuffer <DDR> <301056> -> memref<1x3x224x224xf16, @DDR>
        %3 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x1000xf16, @DDR>
        %4 = VPURT.DeclareBuffer <DDR> <602112> -> memref<1x3x224x224xf16, @DDR>
        %5 = VPURT.DeclareBuffer <DDR> <2048> -> memref<1x1000xf16, @DDR>
        %6 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%6 : !VPURT.Barrier) {
            %13 = func.call @outline1(%arg0, %0) : (memref<3x3x224x224xf16, @DDR>, memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        }
        %7 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%6 : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
            %13 = func.call @outline2(%arg0, %2) : (memref<3x3x224x224xf16, @DDR>, memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        }
        %8 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%7 : !VPURT.Barrier) updates(%8 : !VPURT.Barrier) {
            %13 = func.call @outline3(%arg0, %4) : (memref<3x3x224x224xf16, @DDR>, memref<1x3x224x224xf16, @DDR>) -> memref<1x3x224x224xf16, @DDR>
        }
        %9 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%8 : !VPURT.Barrier) updates(%9 : !VPURT.Barrier) {
            %13 = func.call @batching(%0, %1) : (memref<1x3x224x224xf16, @DDR>, memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        }
        %10 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%9 : !VPURT.Barrier) updates(%10 : !VPURT.Barrier) {
            %13 = func.call @batching(%2, %3) : (memref<1x3x224x224xf16, @DDR>, memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        }
        %11 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%10 : !VPURT.Barrier) updates(%11 : !VPURT.Barrier) {
            %13 = func.call @batching(%4, %5) : (memref<1x3x224x224xf16, @DDR>, memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
        }
        %12 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%11 : !VPURT.Barrier) {
            %13 = func.call @outline4(%1, %3, %5, %arg1) : (memref<1x1000xf16, @DDR>, memref<1x1000xf16, @DDR>, memref<1x1000xf16, @DDR>, memref<3x1000xf16, @DDR>) -> memref<3x1000xf16, @DDR>
        }
        return %arg1 : memref<3x1000xf16, @DDR>
    }

    //CHECK: [[ARG_0:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<3x3x224x224xf16, @DDR>
    //CHECK: [[ARG_1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<3x1000xf16, @DDR>
    //CHECK: [[VAR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x224x224xf16, @DDR>
}
