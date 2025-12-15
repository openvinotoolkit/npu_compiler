//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TwoAxisTilingConsiderDistanceSiblingSubview(%arg0: memref<1x1024x256xf16, @DDR>, %arg1: memref<1x1024x256xf16, @DDR>) {
    %cst_0 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[0, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_1 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[128, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_2 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[256, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_3 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[384, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_4 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[512, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_5 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[640, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_6 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[768, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_7 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[896, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_8 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[1024, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_9 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[1152, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_10 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[1280, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]
    %cst_11 = const.Declare memref<128x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1536x256x1x1xf16, {order = #NHWC}>, [#const.SubView<[1408, 0, 0, 0], [128, 256, 1, 1]>, #const.Sparsify<false>]

    %0 = VPUIP.GenericReshape inputs(%arg0 : memref<1x1024x256xf16, @DDR>) -> memref<1024x256x1x1xf16, @DDR>
    %1 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>} inputs(%0 : memref<1024x256x1x1xf16, @DDR>) -> memref<1x256x1024x1xf16, #NHWC, @DDR>
    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x256x1024x1xf16, #NHWC, @DDR>) -> memref<1x256x256x4xf16, #NHWC, @DDR>


    //    ACT-COPY1   WEIGHT-COPY1     ACT-COPY24   WEIGHT-COPY24
    //         \      /                     \      /
    //           NCE1              ...        NCE24
    //
    //    =>
    //
    //    WEIGHT-COPY1  ACT-COPY1  WEIGHT-COPY6
    //          \     /       \    /
    //            NCE1   ...   NCE6
    //
    //    WEIGHT-COPY7  ACT-COPY2  WEIGHT-COPY12
    //          \     /       \    /
    //            NCE7   ...   NCE12
    //
    //    WEIGHT-COPY13 ACT-COPY1  WEIGHT-COPY18
    //          \     /       \    /
    //            NCE13   ...   NCE18
    //
    //    WEIGHT-COPY19 ACT-COPY2  WEIGHT-COPY24
    //          \     /       \    /
    //            NCE19   ...   NCE24


    // the first set of tiles
    %3 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloc = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %4 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_1 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPUIP.Copy inputs(%3 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_1 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_2 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%5 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%4 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%5 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_2 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_2 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %7 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %7 will be fused to %3
    %alloc_3 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %8 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_3 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_4 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %9 = VPUIP.Copy inputs(%7 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_4 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_5 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %10 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%9 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%8 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%9 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_5 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_5 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %11 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %11 will be fused to %3
    %alloc_6 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %12 = VPUIP.Copy inputs(%cst_2 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_6 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_7 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %13 = VPUIP.Copy inputs(%11 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_7 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_8 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %14 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%13 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%12 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%13 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_8 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_8 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %15 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %15 will be fused to %3
    %alloc_9 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %16 = VPUIP.Copy inputs(%cst_3 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_9 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_10 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %17 = VPUIP.Copy inputs(%15 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_10 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_11 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %18 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%17 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%16 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%17 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_11 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_11 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %19 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %19 will be fused to %3
    %alloc_12 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %20 = VPUIP.Copy inputs(%cst_4 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_12 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_13 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %21 = VPUIP.Copy inputs(%19 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_13 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_14 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %22 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%21 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%20 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%21 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_14 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_14 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %23 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %23 will be fused to %3
    %alloc_15 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %24 = VPUIP.Copy inputs(%cst_5 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloc_15 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloc_16 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %25 = VPUIP.Copy inputs(%23 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloc_16 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloc_17 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %26 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%25 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%24 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%25 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloc_17 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloc_17 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // the second set of tiles
    %27 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    %alloca_18 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %28 = VPUIP.Copy inputs(%cst_0 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_18 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %28 will not be fused, since it is beyond cost distance
    %alloca_19 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %29 = VPUIP.Copy inputs(%27 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_19 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_20 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %30 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%29 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%28 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%29 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_20 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_20 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %31 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %31 will be fused to %27
    %alloca_21 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %32 = VPUIP.Copy inputs(%cst_1 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_21 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %32 will not be fused, since it is beyond cost distance
    %alloca_22 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %33 = VPUIP.Copy inputs(%31 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_22 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_23 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %34 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%33 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%32 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%33 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_23 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_23 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %35 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %35 will be fused to %27
    %alloca_24 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %36 = VPUIP.Copy inputs(%cst_2 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_24 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %36 will not be fused, since it is beyond cost distance
    %alloca_25 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %37 = VPUIP.Copy inputs(%35 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_25 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_26 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %38 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%37 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%36 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%37 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_26 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_26 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %39 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %39 will be fused to %27
    %alloca_27 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %40 = VPUIP.Copy inputs(%cst_3 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_27 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %40 will not be fused, since it is beyond cost distance
    %alloca_28 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %41 = VPUIP.Copy inputs(%39 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_28 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_29 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %42 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%41 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%40 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%41 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_29 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_29 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %43 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %43 will be fused to %27
    %alloca_30 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %44 = VPUIP.Copy inputs(%cst_4 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_30 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %44 will not be fused, since it is beyond cost distance
    %alloca_31 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %45 = VPUIP.Copy inputs(%43 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_31 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_32 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %46 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%45 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%44 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%45 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_32 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_32 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %47 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %47 will be fused to %27
    %alloca_33 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %48 = VPUIP.Copy inputs(%cst_5 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_33 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %48 will not be fused, since it is beyond cost distance
    %alloca_34 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %49 = VPUIP.Copy inputs(%47 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_34 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_35 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %50 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%49 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%48 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%49 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_35 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_35 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // the third set of tiles
    %51 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %51 will be fused to %3
    %alloca_36 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %52 = VPUIP.Copy inputs(%cst_6 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_36 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloca_37 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %53 = VPUIP.Copy inputs(%51 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_37 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_38 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %54 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%53 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%52 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%53 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_38 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_38 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %55 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %55 will be fused to %3
    %alloca_39 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %56 = VPUIP.Copy inputs(%cst_7 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_39 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloca_40 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %57 = VPUIP.Copy inputs(%55 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_40 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_41 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %58 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%57 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%56 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%57 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_41 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_41 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %59 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %59 will be fused to %3
    %alloca_42 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %60 = VPUIP.Copy inputs(%cst_8 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_42 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloca_43 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %61 = VPUIP.Copy inputs(%59 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_43 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_44 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %62 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%61 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%60 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%61 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_44 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_44 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %63 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %63 will be fused to %3
    %alloca_45 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %64 = VPUIP.Copy inputs(%cst_9 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_45 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloca_46 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %65 = VPUIP.Copy inputs(%63 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_46 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_47 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %66 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%65 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%64 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%65 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_47 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_47 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %67 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %67 will be fused to %3
    %alloca_48 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %68 = VPUIP.Copy inputs(%cst_10 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_48 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloca_49 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %69 = VPUIP.Copy inputs(%67 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_49 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_50 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %70 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%69 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%68 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%69 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_50 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_50 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %71 = VPUIP.SubView %2 [0, 0, 0, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %71 will be fused to %3
    %alloca_51 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %72 = VPUIP.Copy inputs(%cst_11 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_51 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %alloca_52 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %73 = VPUIP.Copy inputs(%71 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_52 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_53 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %74 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%73 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%72 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%73 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_53 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_53 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    // the fourth set of tiles
    %75 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %75 will be fused to %27
    %alloca_54 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %76 = VPUIP.Copy inputs(%cst_6 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_54 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %76 will not be fused, since it is beyond cost distance
    %alloca_55 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %77 = VPUIP.Copy inputs(%75 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_55 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_56 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %78 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%77 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%76 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%77 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_56 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_56 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %79 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %79 will be fused to %27
    %alloca_57 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %80 = VPUIP.Copy inputs(%cst_7 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_57 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %80 will not be fused, since it is beyond cost distance
    %alloca_58 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %81 = VPUIP.Copy inputs(%79 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_58 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_59 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %82 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%81 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%80 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%81 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_59 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_59 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
        DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %83 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %83 will be fused to %27
    %alloca_60 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %84 = VPUIP.Copy inputs(%cst_8 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_60 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %84 will not be fused, since it is beyond cost distance
    %alloca_61 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %85 = VPUIP.Copy inputs(%83 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_61 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_62 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %86 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%85 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%84 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%85 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_62 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_62 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %87 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %87 will be fused to %27
    %alloca_63 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %88 = VPUIP.Copy inputs(%cst_9 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_63 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %88 will not be fused, since it is beyond cost distance
    %alloca_64 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %89 = VPUIP.Copy inputs(%87 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_64 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_65 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %90 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%89 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%88 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%89 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_65 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_65 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %91 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %91 will be fused to %27
    %alloca_66 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %92 = VPUIP.Copy inputs(%cst_10 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_66 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %92 will not be fused, since it is beyond cost distance
    %alloca_67 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %93 = VPUIP.Copy inputs(%91 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_67 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_68 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %94 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%93 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%92 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%93 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_68 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_68 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    %95 = VPUIP.SubView %2 [0, 0, 128, 0] [1, 256, 128, 4] : memref<1x256x256x4xf16, #NHWC, @DDR> to memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>
    // %95 will be fused to %27
    %alloca_69 = memref.alloc() : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %96 = VPUIP.Copy inputs(%cst_11 : memref<128x256x1x1xf16, {order = #NHWC}>) outputs(%alloca_69 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // %96 will not be fused, since it is beyond cost distance
    %alloca_70 = memref.alloc() : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %97 = VPUIP.Copy inputs(%95 : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs(%alloca_70 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %alloca_71 = memref.alloc() : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>
    %98 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
    input(%97 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) weights(%96 : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) parent_input(%97 : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%alloca_71 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) outputs(%alloca_71 : memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x128x128x4xf16, #NHWC, [@CMX_NN, 0]> variants : {
    DPUTask {inEnd = [3, 127, 255], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [3, 127, 127], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }

    return
    // CHECK: [[COPY_WEIGHT_1:%.+]] = VPUIP.Copy inputs([[CST_1:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_1:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_ACT_1:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_2:%.+]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NCE_1:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_1]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_2:%.+]] = VPUIP.Copy inputs([[CST_2:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_2:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_2:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_2]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_3:%.+]] = VPUIP.Copy inputs([[CST_3:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_3:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_3:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_3]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_4:%.+]] = VPUIP.Copy inputs([[CST_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_4:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_4]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_5:%.+]] = VPUIP.Copy inputs([[CST_5:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_5:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_5:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_5]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_6:%.+]] = VPUIP.Copy inputs([[CST_6:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_6:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_6:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_6]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[COPY_WEIGHT_7:%.+]] = VPUIP.Copy inputs([[CST_1:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_7:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[COPY_ACT_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_2:%.+]] : memref<1x256x128x4xf16, {order = #NHWC, strides = [262144, 1, 1024, 256]}, @DDR>) outputs([[ALLOC_2:%.+]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK: [[NCE_7:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_7]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_8:%.+]] = VPUIP.Copy inputs([[CST_2:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_8:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_8:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_8]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_9:%.+]] = VPUIP.Copy inputs([[CST_3:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_9:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_9]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_10:%.+]] = VPUIP.Copy inputs([[CST_4:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_10:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_10:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_10]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_11:%.+]] = VPUIP.Copy inputs([[CST_5:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_11:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_11:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_11]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_12:%.+]] = VPUIP.Copy inputs([[CST_6:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_12:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_12:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_12]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[COPY_WEIGHT_13:%.+]] = VPUIP.Copy inputs([[CST_7:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_13:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_13:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_13]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_14:%.+]] = VPUIP.Copy inputs([[CST_8:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_14:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_14:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_14]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_15:%.+]] = VPUIP.Copy inputs([[CST_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_15:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_15:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_15]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_16:%.+]] = VPUIP.Copy inputs([[CST_10:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_16:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_16:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_16]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_17:%.+]] = VPUIP.Copy inputs([[CST_11:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_17:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_17:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_17]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_18:%.+]] = VPUIP.Copy inputs([[CST_12:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_18:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_18:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_1]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_18]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK: [[COPY_WEIGHT_19:%.+]] = VPUIP.Copy inputs([[CST_7:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_19:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_19:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_19]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_20:%.+]] = VPUIP.Copy inputs([[CST_8:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_20:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_20:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_20]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_21:%.+]] = VPUIP.Copy inputs([[CST_9:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_21:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_21:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_21]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_22:%.+]] = VPUIP.Copy inputs([[CST_10:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_22:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_22:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_22]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_23:%.+]] = VPUIP.Copy inputs([[CST_11:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_23:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_23:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_23]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK: [[COPY_WEIGHT_24:%.+]] = VPUIP.Copy inputs([[CST_12:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}>) outputs([[ALLOC_24:%.+]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>) -> memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK: [[NCE_24:%.+]] = VPUIP.NCEClusterTask
    // CHECK:     input([[COPY_ACT_2]] : memref<1x256x128x4xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK:     weights([[COPY_WEIGHT_24]] : memref<128x256x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
}
