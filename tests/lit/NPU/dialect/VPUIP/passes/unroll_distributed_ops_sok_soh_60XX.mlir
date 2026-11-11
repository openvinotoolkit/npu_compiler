//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --unroll-distributed-ops --canonicalize  %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x32x16x16xf16, #NHWC, @CMX_NN, {
        mode = "DUPLICATED",
        num_clusters = 2 : i64,
        uniform_distributed_segments,
        compute_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]],
        compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
        memory_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!WeightsDistributed = !VPUIP.DistributedBuffer<
    96x32x1x1xf16, #NHWC, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [2, 1, 1, 1],
        num_clusters = 2 : i64,
        alignment = [16, 1, 1, 1],
        uniform_distributed_segments,
        compute_shapes = [[48, 32, 1, 1], [48, 32, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]],
        memory_shapes = [[48, 32, 1, 1], [48, 32, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]]
}>

!NewWeightsTablesDistributed = !VPUIP.DistributedBuffer<
    96x1x1x1xf32, #NCHW, @CMX_NN, {
        mode = "SEGMENTED",
        num_tiles = [2, 1, 1, 1],
        num_clusters = 2 : i64,
        alignment = [16, 1, 1, 1],
        uniform_distributed_segments,
        compute_shapes = [[48, 1, 1, 1], [48, 1, 1, 1]],
        compute_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]],
        memory_shapes = [[48, 1, 1, 1], [48, 1, 1, 1]],
        memory_offsets = [[0, 0, 0, 0], [48, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x96x16x16xf16, #NHWC, @CMX_NN, {
        mode = "OVERLAPPED|SEGMENTED",
        num_tiles = [1, 2, 1, 1],
        num_clusters = 2 : i64,
        alignment = [1, 16, 1, 1],
        uniform_distributed_segments,
        compute_shapes = [[1, 48, 16, 16], [1, 48, 16, 16]],
        compute_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]],
        memory_shapes = [[1, 96, 8, 16], [1, 96, 8, 16]],
        memory_offsets = [[0, 0, 0, 0], [0, 0, 8, 0]],
        memory_num_tiles = [1, 1, 2, 1]
}>

// CHECK-LABEL: @UnrollNCEConv
module @UnrollDistributedOpsWithLocalRegion attributes {config.platform = #config.platform<NPU6010>} {
    func.func @UnrollNCEConv(%arg0: memref<1x16x16x32xf16, @DDR>, %arg1: memref<1x16x16x96xf16, @DDR>) -> memref<1x16x16x96xf16, @DDR> {
    %cst = const.Declare memref<96x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<96x1x1x1xf32> = dense<1.000000e+00> : tensor<96x1x1x1xf32>
    %cst_1 = const.Declare memref<96x1x1x1xf32> = dense<0.000000e+00> : tensor<96x1x1x1xf32>
    %0 = VPURT.DeclareBuffer <CMX_NN> <24576> -> !InputDistributed
    %1 = VPURT.DeclareBuffer <CMX_NN> <40960> -> !WeightsDistributed
    %2 = VPURT.DeclareBuffer <CMX_NN> <44032> -> !NewWeightsTablesDistributed
    %3 = VPURT.DeclareBuffer <CMX_NN> <44224> -> !NewWeightsTablesDistributed
    %4 = VPURT.DeclareBuffer <CMX_NN> <0> -> !OutputDistributed
    %5 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %6 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
    %7 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %8 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %9 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %10 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %11 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x96x16x16xf16, {order = #NHWC}, @DDR>
    VPURT.Task updates(%5 : !VPURT.Barrier)  {
        %12 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%6 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) outputs(%0 : !InputDistributed) -> !InputDistributed
    }
    VPURT.Task updates(%7 : !VPURT.Barrier)  {
        %12 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst : memref<96x32x1x1xf16, {order = #NHWC}>) outputs(%1 : !WeightsDistributed) -> !WeightsDistributed
    }
    VPURT.Task updates(%8 : !VPURT.Barrier)  {
        %12 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst_0 : memref<96x1x1x1xf32>) outputs(%2 : !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }
    VPURT.Task updates(%9 : !VPURT.Barrier)  {
        %12 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%cst_1 : memref<96x1x1x1xf32>) outputs(%3 : !NewWeightsTablesDistributed) -> !NewWeightsTablesDistributed
    }
    VPURT.Task waits(%5, %7, %8, %9 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%10 : !VPURT.Barrier)  {
        %12 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1048 : i64} <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}>
        input(%0 : !InputDistributed)
        weights(%1 : !WeightsDistributed)
        weight_table_scale(%2 : !NewWeightsTablesDistributed)
        weight_table_bias(%3 : !NewWeightsTablesDistributed)
        parent_input(%0 : !InputDistributed)
        parent_output(%4 : !OutputDistributed)
        outputs(%4 : !OutputDistributed) -> !OutputDistributed variants : {
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 47], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 95], outStart = [0, 0, 48], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
        }
    }
    VPURT.Task waits(%10 : !VPURT.Barrier)  {
        %12 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%4 : !OutputDistributed) outputs(%11 : memref<1x96x16x16xf16, {order = #NHWC}, @DDR>) -> memref<1x96x16x16xf16, {order = #NHWC}, @DDR>
    }
    return %arg1 : memref<1x16x16x96xf16, @DDR>

    // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare memref<48x32x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare memref<48x32x1x1xf16, {order = #NHWC}>
    // CHECK:       [[CST_WEIGHTS_TABLE_SCALE_1:%.+]] = const.Declare memref<48x1x1x1xf32>
    // CHECK:       [[CST_WEIGHTS_TABLE_SCALE_0:%.+]] = const.Declare memref<48x1x1x1xf32>
    // CHECK:       [[CST_WEIGHTS_TABLE_BIAS_1:%.+]] = const.Declare memref<48x1x1x1xf32>
    // CHECK:       [[CST_WEIGHTS_TABLE_BIAS_0:%.+]] = const.Declare memref<48x1x1x1xf32>

    // CHECK:       [[INPUT_BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <24576> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[INPUT_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <24576> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:       [[INPUT_BUF_DISTRIBUTED:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <24576> -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64

    // CHECK:       [[WEIGHTS_BUF_0_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF_1_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <40960> -> memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    // CHECK:       [[WEIGHTS_BUF_0_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <40960> -> memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[WEIGHTS_BUF_1_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <40960> -> memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>

    // CHECK:       [[OUTPUT_BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x96x8x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:       [[OUTPUT_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x96x8x16xf16, {order = #NHWC}, [@CMX_NN, 1]>

    // CHECK:       [[ITI_BUF_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:      1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
    // CHECK-NEXT:      inwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:      ],
    // CHECK-NEXT:      outwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:          ]>
    // CHECK-NEXT:      ]>

    // CHECK:       [[ITI_BUF_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <-24576> -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:      1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
    // CHECK-NEXT:      inwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:      ],
    // CHECK-NEXT:      outwardHaloRegions = [
    // CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:          ]>
    // CHECK-NEXT:      ]>

    // CHECK:       [[BAR_0:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[NETWORK_INPUT:%.+]] = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BAR_1:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR_3:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[BAR_4:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:       [[NETWORK_OUTPUT_0:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x96x8x16xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[NETWORK_OUTPUT_1:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <24576> -> memref<1x96x8x16xf16, {order = #NHWC}, @DDR>

    // CHECK:       VPURT.Task updates([[BAR_0]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NNDMA
    // CHECK-SAME:          inputs([[NETWORK_INPUT]] : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME:          outputs([[INPUT_BUF_DISTRIBUTED]] : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN

    // CHECK:       VPURT.Task updates([[BAR_1]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:          inputs([[CST_WEIGHTS_0]] : memref<48x32x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:          outputs([[WEIGHTS_BUF_0_1]] : memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task updates([[BAR_1]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK-SAME:          inputs([[CST_WEIGHTS_1]] : memref<48x32x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:          outputs([[WEIGHTS_BUF_1_1]] : memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)

    // CHECK:       VPURT.Task waits([[BAR_0]], [[BAR_1]], [[BAR_2]], [[BAR_3]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates([[BAR_4]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NCEClusterTask
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          kernel_size = [1, 1]
    // CHECK-SAME:          kernel_strides = [1, 1]
    // CHECK-SAME:          local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 7 : i64>
    // CHECK-SAME:          mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[INPUT_BUF_0]] : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_BUF_0_0]] : memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:           parent_output([[ITI_BUF_0]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           output_ITI_buff([[ITI_BUF_1]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           outputs([[ITI_BUF_0]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           DPUTask {cluster_id = 0 : i64, inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 47], outStart = [0, 0, 0]

    // CHECK:       VPURT.Task waits([[BAR_0]], [[BAR_1]], [[BAR_2]], [[BAR_3]] : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates([[BAR_4]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NCEClusterTask
    // CHECK-SAME:          kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:          kernel_size = [1, 1]
    // CHECK-SAME:          kernel_strides = [1, 1]
    // CHECK-SAME:          local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 8 : i64, yEnd = 15 : i64>
    // CHECK-SAME:          mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:      input([[INPUT_BUF_1]] : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK-SAME:      weights([[WEIGHTS_BUF_1_0]] : memref<48x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK:           parent_output([[ITI_BUF_1]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           output_ITI_buff([[ITI_BUF_0]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           outputs([[ITI_BUF_1]] : !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           -> !VPUIP.ITIBuffer<
    // CHECK-NEXT:          1x96x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
    // CHECK-NEXT:          inwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
    // CHECK-NEXT:          ],
    // CHECK-NEXT:          outwardHaloRegions = [
    // CHECK-NEXT:              #VPUIP.OutwardHaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 48, 8, 16], offset = [0, 48, 0, 0], cluster_id = 0 : i64>
    // CHECK-NEXT:              ]>
    // CHECK:           DPUTask {cluster_id = 1 : i64, inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 95], outStart = [0, 0, 48]

    // CHECK:       VPURT.Task waits([[BAR_4]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NNDMA <{port = 0 : i64}>
    // CHECK-SAME:          inputs([[OUTPUT_BUF_0]] : memref<1x96x8x16xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:          outputs([[NETWORK_OUTPUT_0]] : memref<1x96x8x16xf16, {order = #NHWC}, @DDR>)

    // CHECK:       VPURT.Task waits([[BAR_4]] : !VPURT.Barrier)
    // CHECK:           VPUIP.NNDMA <{port = 1 : i64}>
    // CHECK-SAME:          inputs([[OUTPUT_BUF_1]] : memref<1x96x8x16xf16, {order = #NHWC}, [@CMX_NN, 1]>)
    // CHECK-SAME:          outputs([[NETWORK_OUTPUT_1]] : memref<1x96x8x16xf16, {order = #NHWC}, @DDR>)

    }
}
