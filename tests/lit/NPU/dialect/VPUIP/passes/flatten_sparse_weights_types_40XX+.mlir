//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --flatten-sparse-weights-types %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseConvWeightsWithCompressCandidate(%arg0: memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> {
  %cst_weights = const.Declare memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>
    = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
  %cst_weights_sm = const.Declare memref<64x1x1x128xi1> = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

  %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %bar2 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <6688> -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <576>-> memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <4672>-> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <DDR> <0> -> memref<64x32x1x1xf16, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, @DDR>
  %7 = VPURT.DeclareBuffer <CMX_NN> [0] <7840> -> memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>

  VPURT.Task updates(%bar0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %8 = VPUIP.NNDMA inputs(%cst_weights : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>)
                                      outputs(%3 : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%bar2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %8 = VPUIP.NNDMA inputs(%cst_weights_sm : memref<64x1x1x128xi1>) outputs(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>) -> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  }
  VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %8 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64} inputs(%3 : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
                                      outputs(%6 : memref<64x32x1x1xf16, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, @DDR>)
            -> memref<64x32x1x1xf16, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, @DDR>
  }
  VPURT.Task waits(%bar1 : !VPURT.Barrier) updates(%bar2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %8 = VPUIP.NNDMA {compress_candidate, spillId = 0 : i64} inputs(%6 : memref<64x32x1x1xf16, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, @DDR>)
                                       outputs(%7 : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%bar2 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %8 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
            input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%7 : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
            weights_sparsity_map(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>)
            parent_input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            parent_output(%2 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            outputs(%2 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
    -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 2, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
  }

  return %arg1 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>

  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare memref<4096x1x1x1xui8, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>,
  // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.Sparsify<true, dense<32> : tensor<64xi64>>]

  // CHECK:       [[WEIGHTS_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>
  // CHECK:       [[WEIGHTS_SPILL_DDR:%.+]] = VPURT.DeclareBuffer <DDR> <0> -> memref<4096x1x1x1xui8, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC}, @DDR>
  // CHECK:       [[WEIGHTS_CMX_DENSE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <7840> -> memref<64x32x1x1xf16,
  // CHECK-SAME:      {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>
  // CHECK:       [[WEIGHTS_SPILL_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <7840> -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NNDMA inputs([[CST_WEIGHTS]] : memref<4096x1x1x1xui8, {order = #NHWC}>)
  // CHECK-SAME:                                   outputs([[WEIGHTS_CMX]] : memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK-SAME:        -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NNDMA {compress_candidate, spillId = 0 : i64}
  // CHECK-SAME:          inputs([[WEIGHTS_CMX]] : memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK-SAME:          outputs([[WEIGHTS_SPILL_DDR]] : memref<4096x1x1x1xui8, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC}, @DDR>)
  // CHECK-SAME:        -> memref<4096x1x1x1xui8, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC}, @DDR>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NNDMA {compress_candidate, spillId = 0 : i64}
  // CHECK-SAME:          inputs([[WEIGHTS_SPILL_DDR]] : memref<4096x1x1x1xui8, {allocSize = 4192 : i64, compression = #VPUIP.Compression<CompressionCandidate>, order = #NHWC}, @DDR>)
  // CHECK-SAME:          outputs([[WEIGHTS_SPILL_CMX]] : memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK-SAME:        -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>


  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NCEClusterTask
  // CHECK-SAME:          weights([[WEIGHTS_CMX_DENSE]] : memref<64x32x1x1xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>)
}
