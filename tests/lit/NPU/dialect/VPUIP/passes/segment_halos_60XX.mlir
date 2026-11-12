//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --segment-halos %s | FileCheck %s
// REQUIRES: platform-NPU6010


#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoNCEClusterTasksK2H
module @TwoNCEClusterTasksK2H {
  func.func @NCEConv(%arg0: memref<1x16x16x32xf16, @DDR>, %arg1: memref<1x16x16x64xf16, @DDR>) -> memref<1x16x16x64xf16, @DDR> {
    %cst = const.Declare memref<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [32, 32, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<32x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [32, 32, 1, 1]>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare memref<32x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>, [#const.SubView<[32, 0, 0, 0], [32, 1, 1, 1]>]
    %cst_2 = const.Declare memref<32x1x1x1xf32> = dense<1.000000e+00> : tensor<64x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [32, 1, 1, 1]>]
    %cst_3 = const.Declare memref<32x1x1x1xf32> = dense<0.000000e+00> : tensor<64x1x1x1xf32>, [#const.SubView<[32, 0, 0, 0], [32, 1, 1, 1]>]
    %cst_4 = const.Declare memref<32x1x1x1xf32> = dense<0.000000e+00> : tensor<64x1x1x1xf32>, [#const.SubView<[0, 0, 0, 0], [32, 1, 1, 1]>]
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <16384> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [1] <16384> -> memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %2 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <16384> -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_num_tiles = [1, 1, 2, 1]}>
    %3 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <32768> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [1] <32768> -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    %8 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
    %9 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    %10 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
    %11 = VPURT.DeclareBuffer <CMX_NN> [0] <34944> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer <CMX_NN> [1] <34944> -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer <CMX_NN> [0] <34944> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer <CMX_NN> [1] <34944> -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
    %15 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x64x8x16xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %16 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x64x8x16xf16, {order = #NHWC}, [@CMX_NN, 1]>
    %17 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !VPUIP.ITIBuffer<
        1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
            ]>
    ]>
    %18 = VPURT.DeclareBuffer <CMX_NN> [1] <-16384> -> !VPUIP.ITIBuffer<
        1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
            ]>
    ]>
    %19 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %20 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x32x16x16xf16, {order = #NHWC}, @DDR>
    %21 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %22 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %23 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %24 = VPURT.DeclareVirtualBarrier  -> !VPURT.Barrier
    %25 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<1x64x8x16xf16, {order = #NHWC}, @DDR>
    %26 = VPURT.DeclareBuffer <NetworkOutput> [0] <16384> -> memref<1x64x8x16xf16, {order = #NHWC}, @DDR>
    VPURT.Task updates(%19 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%20 : memref<1x32x16x16xf16, {order = #NHWC}, @DDR>) outputs(%2 : !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_num_tiles = [1, 1, 2, 1]}>) -> !VPUIP.DistributedBuffer<1x32x16x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments, compute_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 32, 16, 16], [1, 32, 16, 16]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]], memory_num_tiles = [1, 1, 2, 1]}>
    }
    VPURT.Task updates(%21 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<32x32x1x1xf16, {order = #NHWC}>) outputs(%5 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%21 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst : memref<32x32x1x1xf16, {order = #NHWC}>) outputs(%6 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>
    }
    VPURT.Task updates(%22 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_2 : memref<32x1x1x1xf32>) outputs(%9 : memref<32x1x1x1xf32, [@CMX_NN, 0]>) -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%22 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_1 : memref<32x1x1x1xf32>) outputs(%10 : memref<32x1x1x1xf32, [@CMX_NN, 1]>) -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
    }
    VPURT.Task updates(%23 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_4 : memref<32x1x1x1xf32>) outputs(%13 : memref<32x1x1x1xf32, [@CMX_NN, 0]>) -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    }
    VPURT.Task updates(%23 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 1 : i64} inputs(%cst_3 : memref<32x1x1x1xf32>) outputs(%14 : memref<32x1x1x1xf32, [@CMX_NN, 1]>) -> memref<32x1x1x1xf32, [@CMX_NN, 1]>
    }
    VPURT.Task waits(%19, %21, %22, %23 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%24 : !VPURT.Barrier)  {
      %27 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 7 : i64>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%0 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) weights(%3 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) weight_table_scale(%7 : memref<32x1x1x1xf32, [@CMX_NN, 0]>) weight_table_bias(%11 : memref<32x1x1x1xf32, [@CMX_NN, 0]>) parent_input(%0 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_output(%17 : !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
              ]>
      ]>) output_ITI_buff(%18 : !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
              ]>
      ]>) outputs(%17 : !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
              ]>
      ]>) -> !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
              ]>
      ]> variants : { // Workloads split over K
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 15, 15], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 15], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 0 : i64, inEnd = [15, 15, 31], inStart = [0, 0, 16], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 31], outStart = [0, 0, 16], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
      }
    }
    VPURT.Task waits(%19, %21, %22, %23 : !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier, !VPURT.Barrier) updates(%24 : !VPURT.Barrier)  {
      %27 = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 8 : i64, yEnd = 15 : i64>, mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}> input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) weights(%4 : memref<32x32x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>) weight_table_scale(%8 : memref<32x1x1x1xf32, [@CMX_NN, 1]>) weight_table_bias(%12 : memref<32x1x1x1xf32, [@CMX_NN, 1]>) parent_input(%1 : memref<1x32x16x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) parent_output(%18 : !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
              ]>
      ]>) output_ITI_buff(%17 : !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
              ]>
      ]>) outputs(%18 : !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
              ]>
      ]>) -> !VPUIP.ITIBuffer<
          1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
          inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
          ],
          outwardHaloRegions = [
              #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
              ]>
      ]> variants : { // Workloads split over W
        DPUTask {cluster_id = 1 : i64, inEnd = [7, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [7, 15, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, inEnd = [15, 15, 31], inStart = [8, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_8x16>, outEnd = [15, 15, 63], outStart = [8, 0, 32], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      } PPE : {
        PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>}
      }
    }
    VPURT.Task waits(%24 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 0 : i64} inputs(%15 : memref<1x64x8x16xf16, {order = #NHWC}, [@CMX_NN, 0]>) outputs(%25 : memref<1x64x8x16xf16, {order = #NHWC}, @DDR>) -> memref<1x64x8x16xf16, {order = #NHWC}, @DDR>
    }
    VPURT.Task waits(%24 : !VPURT.Barrier)  {
      %27 = VPUIP.NNDMA {port = 1 : i64} inputs(%16 : memref<1x64x8x16xf16, {order = #NHWC}, [@CMX_NN, 1]>) outputs(%26 : memref<1x64x8x16xf16, {order = #NHWC}, @DDR>) -> memref<1x64x8x16xf16, {order = #NHWC}, @DDR>
    }
    return %arg1 : memref<1x16x16x64xf16, @DDR>
  }
}

// CHECK:       [[OUT_CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 0],
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 8], offset = [0, 32, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 8], offset = [0, 32, 0, 8], cluster_id = 0 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:              shape = [1, 16, 8, 16],
// CHECK-SAME:              offset = [0, 0, 8, 0],
// CHECK-SAME:              cluster_id = 0 : i64,
// CHECK-SAME:              inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>
// CHECK-NEXT:              ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:              shape = [1, 16, 8, 16],
// CHECK-SAME:              offset = [0, 16, 8, 0],
// CHECK-SAME:              cluster_id = 0 : i64,
// CHECK-SAME:              inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 8, 0], cluster_id = 1 : i64>
// CHECK-NEXT:              ]>

// CHECK:       [[OUT_CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <-16384> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x64x16x16xf16, {order = #NHWC}, [@CMX_NN, 1],
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 8, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 8, 0], cluster_id = 1 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:              shape = [1, 32, 8, 8],
// CHECK-SAME:              offset = [0, 32, 0, 0],
// CHECK-SAME:              cluster_id = 1 : i64,
// CHECK-SAME:              inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 8], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
// CHECK-NEXT:              ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:              shape = [1, 32, 8, 8],
// CHECK-SAME:              offset = [0, 32, 0, 8],
// CHECK-SAME:              cluster_id = 1 : i64,
// CHECK-SAME:              inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 8], offset = [0, 32, 0, 8], cluster_id = 0 : i64>
// CHECK-NEXT:              ]>

// CHECK:       VPUIP.NCEClusterTask <{
// CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:      kernel_size = [1, 1],
// CHECK-SAME:      kernel_strides = [1, 1],
// CHECK-SAME:      local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 7 : i64>,
// CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
// CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>}
// CHECK:       output_ITI_buff([[OUT_CMX1]]
// CHECK:       outputs([[OUT_CMX0]]

// CHECK:       VPUIP.NCEClusterTask <{
// CHECK-SAME:      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
// CHECK-SAME:      kernel_size = [1, 1],
// CHECK-SAME:      kernel_strides = [1, 1],
// CHECK-SAME:      local_region = #VPUIP.LocalRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 8 : i64, yEnd = 15 : i64>,
// CHECK-SAME:      mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
// CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>}
// CHECK:       output_ITI_buff([[OUT_CMX0]]
// CHECK:       outputs([[OUT_CMX1]]
