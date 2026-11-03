//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true compilation-mode=DefaultHW" --compute-task-stripping="dpu-dry-run=strip shave-dry-run=true" %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8<0:254>:f16:0, {2.3954496608944391E-5:127,1.8652968519315946E-5:127,5.714536651851624E-6:127,9.2288640540415852E-6:127,5.2774985005536418E-5:127,1.0355251041922983E-5:127,4.7608623354453737E-5:127,4.2622483621432085E-5:127,6.3378041184793303E-5:127,9.0411328893946842E-6:127,3.7636343888410434E-5:127,9.1462623415969495E-6:127,3.0472522645484744E-5:127,1.3648806181992955E-6:127,8.4283783679872047E-5:127,5.5778683639886814E-5:127,1.6640490434301182E-5:127,5.4847537063238186E-5:127,9.0531476839320868E-5:127,1.0873389056348425E-5:127,8.4944597379429132E-5:127,5.5928868571604333E-5:127,7.9477865864911412E-5:127,1.5408973994217519E-5:127,2.7033287709153543E-5:127,7.3740801473302161E-6:127,4.5475997324064961E-5:127,4.5415923351377953E-5:127,1.4605484609528789E-5:127,8.4554116556963585E-6:127,1.9478985643762302E-5:127,4.6332051434854825E-6:127,9.6568911094365155E-6:127,1.3648806594488189E-4:127,4.8584825410617617E-6:127,1.0588037686085138E-5:127,9.6493818628506396E-6:127,1.0663130151943897E-5:127,3.8086898683562992E-5:127,4.142100416769193E-5:127,7.5805844284418062E-6:127,5.2684874046505907E-5:127,9.1462623415969495E-6:127,1.327634796382874E-5:127,6.1538275771253692E-6:127,2.9691561000553643E-5:127,5.6079053503321852E-5:127,1.4605484609528789E-5:127,2.8324878121924213E-5:127,6.1695969949557085E-5:127,5.0492174043430117E-5:127,2.3218590443528543E-5:127,1.5026002418337845E-5:127,4.4184406911294294E-5:127,4.0009265809547241E-5:127,2.0289984275036911E-5:127,2.0680465097502461E-5:127,5.5906340831846705E-6:127,1.6054769200602853E-5:127,1.0940972275621308E-5:127,2.3008331539124016E-5:127,3.4182090458907481E-5:127,9.4766691913754922E-6:127,8.9450145330954728E-5:127}>
!qElemType1 = !quant.uniform<i8<-127:127>:f16:0, {2.3954496608944391E-5,1.8652968519315946E-5,5.714536651851624E-6,9.2288640540415852E-6,5.2774985005536418E-5,1.0355251041922983E-5,4.7608623354453737E-5,4.2622483621432085E-5,6.3378041184793303E-5,9.0411328893946842E-6,3.7636343888410434E-5,9.1462623415969495E-6,3.0472522645484744E-5,1.3648806181992955E-6,8.4283783679872047E-5,5.5778683639886814E-5,1.6640490434301182E-5,5.4847537063238186E-5,9.0531476839320868E-5,1.0873389056348425E-5,8.4944597379429132E-5,5.5928868571604333E-5,7.9477865864911412E-5,1.5408973994217519E-5,2.7033287709153543E-5,7.3740801473302161E-6,4.5475997324064961E-5,4.5415923351377953E-5,1.4605484609528789E-5,8.4554116556963585E-6,1.9478985643762302E-5,4.6332051434854825E-6,9.6568911094365155E-6,1.3648806594488189E-4,4.8584825410617617E-6,1.0588037686085138E-5,9.6493818628506396E-6,1.0663130151943897E-5,3.8086898683562992E-5,4.142100416769193E-5,7.5805844284418062E-6,5.2684874046505907E-5,9.1462623415969495E-6,1.327634796382874E-5,6.1538275771253692E-6,2.9691561000553643E-5,5.6079053503321852E-5,1.4605484609528789E-5,2.8324878121924213E-5,6.1695969949557085E-5,5.0492174043430117E-5,2.3218590443528543E-5,1.5026002418337845E-5,4.4184406911294294E-5,4.0009265809547241E-5,2.0289984275036911E-5,2.0680465097502461E-5,5.5906340831846705E-6,1.6054769200602853E-5,1.0940972275621308E-5,2.3008331539124016E-5,3.4182090458907481E-5,9.4766691913754922E-6,8.9450145330954728E-5}>
!qElemType2 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType3 = !quant.uniform<u8:f16, 1.000000e+00:114>

// CHECK-LABEL: @DPUComputeTaskStripping
// CHECK-SAME: ([[ARG_0:%[^:]+]]: memref<1x3x224x224xf16, @DDR>, [[ARG_1:%[^:]+]]: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR>
func.func @DPUComputeTaskStripping(%arg0: memref<1x3x224x224xf16, @DDR>, %arg1: memref<1x64x112x112xf16, @DDR>) -> memref<1x64x112x112xf16, @DDR> {
    // barriers

    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // constants

    %cst = const.Declare memref<64x1x1x160x!qElemType, {order = #NHWC}> = dense<1> : tensor<64x3x7x7xsi8>, [#const.CastElemType<f32>, #const.CastElemType<f16>, #const.CastElemType<si8>, #const.CastElemType<!qElemType1>, #const.CastElemType<si8>, #const.CastElemType<i32>, #const.Add<1.270000e+02 : f64>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>, #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    %cst_0 = const.Declare memref<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[130496, 130496, 130496, 130496], sparsityPtr=16777215 : i64, offsets=[0, 0, 0, 0], weightsTableSize=1024 : i64, weightsElemBitSize=8 : i64>]

    // buffers

    %3 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %4 = VPURT.DeclareBuffer <NetworkInput> [0] <25088> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %5 = VPURT.DeclareBuffer <NetworkInput> [0] <50176> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <75264> -> memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>
    %7 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %8 = VPURT.DeclareBuffer <NetworkOutput> [0] <6272> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %9 = VPURT.DeclareBuffer <NetworkOutput> [0] <12544> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %10 = VPURT.DeclareBuffer <NetworkOutput> [0] <18816> -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 2]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x3x56x224xf16, [@CMX_NN, 3]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>
    %19 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <130496> -> !VPUIP.DistributedBuffer<64x1x1x160x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %24 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <140736> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    %25 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 0]>
    %26 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 1]>
    %27 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 2]>
    %28 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, [@CMX_NN, 3]>
    %29 = VPURT.DeclareBuffer <CMX_NN> [0] <141760> -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %30 = VPURT.DeclareBuffer <CMX_NN> [1] <141760> -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %31 = VPURT.DeclareBuffer <CMX_NN> [2] <141760> -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 2]>
    %32 = VPURT.DeclareBuffer <CMX_NN> [3] <141760> -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 3]>
    %33 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %34 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %35 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 2]>
    %36 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 3]>
    %37 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %38 = VPURT.DeclareBuffer <CMX_NN> [1] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %39 = VPURT.DeclareBuffer <CMX_NN> [2] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 2]>
    %40 = VPURT.DeclareBuffer <CMX_NN> [3] <576> -> memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 3]>
    %41 = VPURT.DeclareBuffer <CMX_NN> [0] <130496> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %42 = VPURT.DeclareBuffer <CMX_NN> [1] <130496> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>
    %43 = VPURT.DeclareBuffer <CMX_NN> [2] <130496> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>
    %44 = VPURT.DeclareBuffer <CMX_NN> [3] <130496> -> memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 3]>
    %45 = VPURT.DeclareBuffer <CMX_NN> [0] <75840> -> memref<1x16x58x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>
    %46 = VPURT.DeclareBuffer <CMX_NN> [1] <75840> -> memref<1x16x61x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>
    %47 = VPURT.DeclareBuffer <CMX_NN> [2] <75840> -> memref<1x16x61x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 2]>
    %48 = VPURT.DeclareBuffer <CMX_NN> [3] <75840> -> memref<1x16x59x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 3]>

    // Multiple tasks to be erased

    //  DMA DMA DMA DMA
    //     \ |  |  /
    //       bar 0
    //     / |  |  \
    //  DPU DPU DPU DPU  DMA DMA
    //     \  \  |  /   /   /
    //      \  \ | /   /   /
    //         bar 1
    //      /  |  |  \
    //    DPU DPU DPU DPU
    //      \  |   | /
    //         bar 2
    //      /  |  |  \
    //    DMA  DMA DMA DMA

    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%3 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%11 : memref<1x3x56x224xf16, [@CMX_NN, 0]>) -> memref<1x3x56x224xf16, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%4 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%12 : memref<1x3x56x224xf16, [@CMX_NN, 1]>) -> memref<1x3x56x224xf16, [@CMX_NN, 1]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%5 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%13 : memref<1x3x56x224xf16, [@CMX_NN, 2]>) -> memref<1x3x56x224xf16, [@CMX_NN, 2]>
    }
    VPURT.Task updates(%bar0 : !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%6 : memref<1x3x56x224xf16, {order = #NCHW, strides = [150528, 50176, 224, 1]}, @DDR>) outputs(%14 : memref<1x3x56x224xf16, [@CMX_NN, 3]>) -> memref<1x3x56x224xf16, [@CMX_NN, 3]>
    }
    VPURT.Task updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<64x1x1x160x!qElemType, {order = #NHWC}>) outputs(%19 : !VPUIP.DistributedBuffer<64x1x1x160x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<64x1x1x160x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160], [64, 1, 1, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    }
    VPURT.Task updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%cst_0 : memref<64x1x1x4xsi32>) outputs(%24 : !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4], [64, 1, 1, 4]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%37 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%33 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%37 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%15 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>) output_ITI_buff(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) outputs(%15 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]> variants : {
        DPUTask {cluster_id = 0 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 53 : i64, yEnd = 55 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -47488 : i64, targetClusters = [1], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 2, 223], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%38 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 1]>) weights(%34 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 1]>) parent_input(%38 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 1]>) parent_output(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) output_ITI_buff(%15, %17 : !VPUIP.ITIBuffer<1x224x4x58x!qElemType2, {order = #NWCH}, [@CMX_NN, 0], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 53], cluster_id = 0 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>]>]>, !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) outputs(%16 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]> variants : {
        DPUTask {cluster_id = 1 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 47488 : i64, targetClusters = [0], targetWidth = 224 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 56 : i64, yEnd = 58 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -50176 : i64, targetClusters = [2], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%39 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 2]>) weights(%35 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 2]>) parent_input(%39 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 2]>) parent_output(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) output_ITI_buff(%16, %18 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 1], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 1 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 56], cluster_id = 0 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 1 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>]>]>, !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) outputs(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]> variants : {
        DPUTask {cluster_id = 2 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 50176 : i64, targetClusters = [1], targetWidth = 224 : i64>, #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 56 : i64, yEnd = 58 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = -50176 : i64, targetClusters = [3], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>}> input(%40 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 3]>) weights(%36 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 3]>) parent_input(%40 : memref<1x224x3x56xf16, {order = #NHWC}, [@CMX_NN, 3]>) parent_output(%18 : !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) output_ITI_buff(%17 : !VPUIP.ITIBuffer<1x224x4x61x!qElemType2, {order = #NWCH}, [@CMX_NN, 2], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 2 : i64>, #VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 1 : i64>]>, #VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 56], cluster_id = 2 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>]>]>) outputs(%18 : !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]>) -> !VPUIP.ITIBuffer<1x224x4x59x!qElemType2, {order = #NWCH}, [@CMX_NN, 3], inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 3], offset = [0, 0, 0, 0], cluster_id = 3 : i64>], outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 3], cluster_id = 3 : i64, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 224, 3, 2], offset = [0, 0, 0, 59], cluster_id = 2 : i64>]>]> variants : {
        DPUTask {cluster_id = 3 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 223 : i64, yStart = 3 : i64, yEnd = 4 : i64, zStart = 0 : i64, zEnd = 2 : i64, targetOffset = 50176 : i64, targetClusters = [2], targetWidth = 224 : i64>], inEnd = [55, 2, 223], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [58, 2, 223], outStart = [3, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}> input(%45 : memref<1x16x58x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>) weights(%41 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%45 : memref<1x16x58x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%29 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%29 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 0]> variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [223, 57, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}> input(%46 : memref<1x16x61x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>) weights(%42 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 1]>) parent_input(%46 : memref<1x16x61x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 1]>) parent_output(%30 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 1]>) outputs(%30 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 1]> variants : {
        DPUTask {cluster_id = 1 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}> input(%47 : memref<1x16x61x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 2]>) weights(%43 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 2]>) parent_input(%47 : memref<1x16x61x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 2]>) parent_output(%31 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 2]>) outputs(%31 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 2]>) -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 2]> variants : {
        DPUTask {cluster_id = 2 : i64, inEnd = [223, 60, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar1: !VPURT.Barrier) updates(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, input_channels_compression, kernel_padding = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>, kernel_size = [7, 7], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}> input(%48 : memref<1x16x59x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 3]>) weights(%44 : memref<64x16x7x7x!qElemType, {order = #NHWC}, [@CMX_NN, 3]>) parent_input(%48 : memref<1x16x59x224x!qElemType3, {order = #NHWC}, [@CMX_NN, 3]>) parent_output(%32 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 3]>) outputs(%32 : memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 3]>) -> memref<1x64x28x112xf16, {order = #NHWC}, [@CMX_NN, 3]> variants : {
        DPUTask {cluster_id = 3 : i64, inEnd = [223, 58, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [111, 27, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 0 : i64, bottom = 2 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
      }
    }
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%25 : memref<1x64x28x112xf16, [@CMX_NN, 0]>) outputs(%7 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%26 : memref<1x64x28x112xf16, [@CMX_NN, 1]>) outputs(%8 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%27 : memref<1x64x28x112xf16, [@CMX_NN, 2]>) outputs(%9 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }
    VPURT.Task waits(%bar2: !VPURT.Barrier) {
      %49 = VPUIP.NNDMA <{port = 1 : i64}> inputs(%28 : memref<1x64x28x112xf16, [@CMX_NN, 3]>) outputs(%10 : memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>) -> memref<1x64x28x112xf16, {order = #NCHW, strides = [802816, 12544, 112, 1]}, @DDR>
    }

    return %arg1 : memref<1x64x112x112xf16, @DDR>


    //  DMA DMA DMA DMA DMA DMA
    //     \  \  |  |  /   /
    //           bar 0
    //        /  |  |  \
    //     DMA  DMA DMA DMA

    // CHECK-NOT:   VPUIP.NCEClusterTask
    // CHECK-NOT:   DPUTask

    // CHECK:       [[BARRIER_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task updates([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA
    // CHECK:       VPURT.Task waits([[BARRIER_0]] : !VPURT.Barrier)
    // CHECK:         VPUIP.NNDMA

    // CHECK:       return [[ARG_1]] : memref<1x64x112x112xf16, @DDR>
}
