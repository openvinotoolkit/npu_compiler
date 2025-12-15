//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --flatten-sparse-weights-types %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SparseConvWeights(%arg0: memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> {
  %cst_weights = const.Declare memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>
    = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
  %cst_weights_sm = const.Declare memref<64x1x1x128xi1> = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

  %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <6688> -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0] <576>-> memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <4672>-> memref<64x1x1x128xi1, [@CMX_NN, 0]>

  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %6 = VPUIP.NNDMA inputs(%cst_weights : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>)
                                      outputs(%3 : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
            -> memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
  }
  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %6 = VPUIP.NNDMA inputs(%cst_weights_sm : memref<64x1x1x128xi1>) outputs(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>) -> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %6 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}
            input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
            weights(%3 : memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>)
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

  // CHECK:       [[WEIGHTS_CMX_DENSE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<64x32x1x1xf16,
  // CHECK-SAME:      {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>
  // CHECK:       [[WEIGHTS_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NNDMA inputs([[CST_WEIGHTS]] : memref<4096x1x1x1xui8, {order = #NHWC}>)
  // CHECK-SAME:                                   outputs([[WEIGHTS_CMX]] : memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>)
  // CHECK-SAME:        -> memref<4096x1x1x1xui8, {order = #NHWC}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NCEClusterTask
  // CHECK-SAME:          weights([[WEIGHTS_CMX_DENSE]] : memref<64x32x1x1xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>)
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedBuffer = !VPUIP.DistributedBuffer<
    64x32x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters = 2,
    compute_shapes = [[64, 32, 1, 1], [64, 32, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[64, 32, 1, 1], [64, 32, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}, #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>>

!CMXWeightsType = memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}, [@CMX_NN, 0]>
!DDRWeightsType = memref<64x32x1x1xf16, {sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>, order = #NHWC}>


func.func @SparseConvWeightsDistributed(%arg0: memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>, %arg1: memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> {
  %cst_weights = const.Declare !DDRWeightsType = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
  %cst_weights_sm = const.Declare memref<64x1x1x128xi1> = dense<1.0> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

  %0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

  %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %2 = VPURT.DeclareBuffer <CMX_NN> [0] <6688> -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>
  %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <576> -> !DistributedBuffer
  %4 = VPURT.DeclareBuffer <CMX_NN> [0] <4672>-> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  %6 = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> !CMXWeightsType

  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NNDMA inputs(%cst_weights : !DDRWeightsType) outputs(%3 : !DistributedBuffer) -> !DistributedBuffer
  }

  VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NNDMA inputs(%cst_weights_sm : memref<64x1x1x128xi1>) outputs(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>)
      -> memref<64x1x1x128xi1, [@CMX_NN, 0]>
  }

  VPURT.Task waits(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false} {
    %7 = VPUIP.NCEClusterTask {
      kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      kernel_size = [1, 1],
      kernel_strides = [1, 1],
      task_type = #VPUIP.nce_task_type<CONV>
     }
      input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
      weights(%6 : !CMXWeightsType)
      weights_sparsity_map(%4 : memref<64x1x1x128xi1, [@CMX_NN, 0]>)
      parent_input(%1 : memref<1x32x3x3xf16, #NHWC, [@CMX_NN, 0]>)
      parent_output(%2 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
      outputs(%2 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>)
        -> memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]> variants : {
      DPUTask {
        cluster_id = 0 : i64,
        mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
        outEnd = [15, 2, 63],
        outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEStub<>}
    }
  }

  return %arg1 : memref<1x64x3x3xf16, #NHWC, [@CMX_NN, 0]>

  // CHECK:       [[CST_WEIGHTS:%.+]] = const.Declare memref<4096x1x1x1xui8, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x1x1xf16>,
  // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.Sparsify<true, dense<32> : tensor<64xi64>>]

  // CHECK:       [[WEIGHTS_CMX:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0, 1] <576>
  // CHECK-SAME:       -> !VPUIP.DistributedBuffer<4096x1x1x1xui8, #NHWC, @CMX_NN,
  // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64,
  // CHECK-SAME{LITERAL}:   compute_shapes = [[4096, 1, 1, 1], [4096, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:   memory_shapes = [[4096, 1, 1, 1], [4096, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}

  // CHECK:       [[WEIGHTS_CMX_DENSE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<64x32x1x1xf16,
  // CHECK-SAME:      {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NNDMA
  // CHECK-SAME:    inputs([[CST_WEIGHTS]] : memref<4096x1x1x1xui8, {order = #NHWC}>)
  // CHECK-SAME:    outputs([[WEIGHTS_CMX]] : !VPUIP.DistributedBuffer<4096x1x1x1xui8, #NHWC, @CMX_NN,
  // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64,
  // CHECK-SAME{LITERAL}:   compute_shapes = [[4096, 1, 1, 1], [4096, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:   memory_shapes = [[4096, 1, 1, 1], [4096, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>)
  // CHECK-SAME:       -> !VPUIP.DistributedBuffer<4096x1x1x1xui8, #NHWC, @CMX_NN,
  // CHECK-SAME:           {mode = "DUPLICATED", num_clusters = 2 : i64,
  // CHECK-SAME{LITERAL}:   compute_shapes = [[4096, 1, 1, 1], [4096, 1, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
  // CHECK-SAME{LITERAL}:   memory_shapes = [[4096, 1, 1, 1], [4096, 1, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

  // CHECK:       VPURT.Task
  // CHECK:           VPUIP.NCEClusterTask
  // CHECK-SAME:          weights([[WEIGHTS_CMX_DENSE]] : memref<64x32x1x1xf16, {order = #NHWC, sparsityCompression = #VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<32> : tensor<64xi64>, alignment = 16 : i64>}, [@CMX_NN, 0]>)
}
