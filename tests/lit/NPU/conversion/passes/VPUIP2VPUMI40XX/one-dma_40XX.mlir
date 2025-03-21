//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-VPUIP-to-VPUMI40XX %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @OneDMAWithoutAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL0:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x2x3x4xf16, @DDR>, outputType = memref<1x2x3x4xf16, @DDR>>){{.*}}-> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}

// -----

module @OneDMAWithAttributes {
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "input_0" : tensor<1x2x3x4xf16>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x2x3x4xf16>
  }
  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    VPURT.Task {
      %0 = VPUIP.NNDMA {is_out_of_order, is_critical, port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL0:.*]] = VPUMI40XX.NNDMA {is_critical, is_out_of_order, port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<1x2x3x4xf16, @DDR>, outputType = memref<1x2x3x4xf16, @DDR>>){{.*}}-> !VPURegMapped.Index<0:0:0>

    return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @OneDMA {
IE.CNNNetwork entryPoint : @UnrollDMAOutput inputsInfo : {
  DataInfo "input_0" : tensor<1x16x16x16xf16>
} outputsInfo : {
  DataInfo "output_0" : tensor<64x32x1x1xf16>
}
func.func @UnrollDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK-DAG: %[[CST:.*]] = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR>

  %3 = VPURT.DeclareBuffer <CMX_NN> [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>

  VPURT.Task attributes {isTrailingSWLayer = false} {
    %16 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>) -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>
  }
  // CHECK: %[[BUFF_TILE_0:.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
  // CHECK: %[[BUFF_TILE_1:.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA0:.*]] = VPUMI40XX.NNDMA {port = 0 : i64} inputs(%[[CST]] : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%[[BUFF_TILE_0]], %[[BUFF_TILE_1]] : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>, memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) start_after(0) clean_after(0) acceleration_mode(<DISABLE>){{.*}}dma_transaction(#VPUMI40XX.NNDMATransaction<inputType = memref<64x32x1x1xf16, #NHWC, @DDR>, outputType = !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments}>>){{.*}}-> !VPURegMapped.Index<0:0:0>

  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}

// -----

IE.CNNNetwork entryPoint : @singleDMADistributeBufferWithNotAdjacentClusters inputsInfo : {
  DataInfo "dummy_input" : tensor<1x50x1x1xf16>
} outputsInfo : {
  DataInfo "dummy_output" : tensor<1x50x1x1xf16>
}
func.func @singleDMADistributeBufferWithNotAdjacentClusters() {
  %3 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
  %4 = VPURT.ConfigureBarrier<2> -> !VPURT.Barrier
  %cst = const.Declare memref<16x32x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR> = dense<0.0> : tensor<32x32x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 32, 1, 1]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
  %5 = VPURT.DeclareBuffer <CMX_NN> [0, 2] <0> -> !VPUIP.DistributedBuffer<16x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK: %[[BUFFER_0:.+]] = VPURT.DeclareBuffer
  // CHECK-SAME: [0]
  // CHECK-NOT: -> !VPUIP.DistributedBuffer
  // CHECK-SAME: -> [[BUFFER_0_TYPE:.+]]
  // CHECK: %[[BUFFER_2:.+]] = VPURT.DeclareBuffer
  // CHECK-SAME: [2]
  // CHECK-NOT: -> !VPUIP.DistributedBuffer
  // CHECK-SAME: -> [[BUFFER_2_TYPE:.+]]
  VPURT.Task waits(%3 : !VPURT.Barrier) updates(%4 : !VPURT.Barrier) {
    %28 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst : memref<16x32x1x1xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, @DDR>) outputs(%5 : !VPUIP.DistributedBuffer<16x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x32x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }
  // CHECK-NOT: VPURT.Task
  // CHECK-NOT: VPUIP.NNDMA
  // CHECK: VPUMI40XX.NNDMA
  // CHECK-SAME: outputs(%[[BUFFER_0]], %[[BUFFER_2]] : [[BUFFER_0_TYPE]], [[BUFFER_2_TYPE]])
  return
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
IE.CNNNetwork entryPoint : @singleDMA inputsInfo : {
  DataInfo "dummy_input" : tensor<1x50x1x1xf16>
} outputsInfo : {
  DataInfo "dummy_output" : tensor<1x50x1x1xf16>
}
func.func @singleDMA() {
    %cst_4 = const.Declare memref<1x3x62x2xf16> = dense<0.000000e+00> : tensor<372xf16>, [#const.Reshape<[1, 3, 62, 2]>, #const.Reorder<#NCHW>]
    // CHECK: %[[INPUT:.+]] = const.Declare memref<[[INPUT_TYPE:.+]]> =
    %250 = VPURT.DeclareBuffer <DDR> <124> -> memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>
    // CHECK: %[[OUTPUT:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE:.+]]

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: %[[BAR0:.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 1
    // CHECK-SAME: producer_count = 0
    // CHECK-SAME: <0, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: %[[BAR1:.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 0
    // CHECK-SAME: producer_count = 1
    // CHECK-SAME: <1, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:1>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %364 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_4 : memref<1x3x62x2xf16>) outputs(%250 : memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>) -> memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>
    }
    // CHECK-NOT: VPURT.Task
    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: VPUMI40XX.NNDMA
    // CHECK-SAME: port = 0
    // CHECK-SAME inputs(%[[INPUT]] : memref<[[INPUT_TYPE]]>)
    // CHECK-SAME: outputs(%[[OUTPUT]] : [[OUTPUT_TYPE]])
    // CHECK-SAME: waits(%[[BAR0]] : !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: updates(%[[BAR1]] : !VPURegMapped.Index<0:0:1>)
    // CHECK-SAME: start_after(0)
    // CHECK-SAME: clean_after(0)
    // CHECK-SAME: acceleration_mode(<DISABLE>)
    // CHECK-SAME: dma_transaction(#VPUMI40XX.NNDMATransaction<
    // CHECK-SAME: inputType = memref<[[INPUT_TYPE]]>
    // CHECK-SAME: outputType = [[OUTPUT_TYPE]]
    // CHECK-SAME: >)
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    return
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
IE.CNNNetwork entryPoint : @doubleDMA inputsInfo : {
  DataInfo "dummy_input" : tensor<1x50x1x1xf16>
} outputsInfo : {
  DataInfo "dummy_output" : tensor<1x50x1x1xf16>
}
func.func @doubleDMA() {
    %cst_4 = const.Declare memref<1x3x62x2xf16> = dense<0.000000e+00> : tensor<372xf16>, [#const.Reshape<[1, 3, 62, 2]>, #const.Reorder<#NCHW>]
    // CHECK: %[[INPUT:.+]] = const.Declare memref<[[INPUT_TYPE:.+]]> =
    %250 = VPURT.DeclareBuffer <DDR> <124> -> memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>
    // CHECK: %[[OUTPUT:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE:.+]]

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: %[[BAR0:.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 1
    // CHECK-SAME: producer_count = 0
    // CHECK-SAME: <0, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    %1 = VPURT.ConfigureBarrier<1> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: %[[BAR1:.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 1
    // CHECK-SAME: producer_count = 1
    // CHECK-SAME: <1, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:1>

    VPURT.Task waits(%0 : !VPURT.Barrier) updates(%1 : !VPURT.Barrier) {
      %364 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_4 : memref<1x3x62x2xf16>) outputs(%250 : memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>) -> memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>
    }
    // CHECK-NOT: VPURT.Task
    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: VPUMI40XX.NNDMA
    // CHECK-SAME: port = 0
    // CHECK-SAME: inputs(%[[INPUT]] : memref<[[INPUT_TYPE]]>)
    // CHECK-SAME: outputs(%[[OUTPUT]] : [[OUTPUT_TYPE]])
    // CHECK-SAME: waits(%[[BAR0]] : !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: updates(%[[BAR1]] : !VPURegMapped.Index<0:0:1>)
    // CHECK-SAME: start_after(0)
    // CHECK-SAME: clean_after(0)
    // CHECK-SAME: acceleration_mode(<DISABLE>)
    // CHECK-SAME: dma_transaction(#VPUMI40XX.NNDMATransaction<
    // CHECK-SAME:    inputType = memref<[[INPUT_TYPE]]>
    // CHECK-SAME:    outputType = [[OUTPUT_TYPE]]
    // CHECK-SAME: >)
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    VPURT.Task waits(%1 : !VPURT.Barrier) {
      %364 = VPUIP.NNDMA {port = 0 : i64} inputs(%cst_4 : memref<1x3x62x2xf16>) outputs(%250 : memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>) -> memref<1x3x62x2xf16, {order = #NCHW, strides = [11904, 3968, 64, 1]}, @DDR>
    }
    // CHECK-NOT: VPURT.Task
    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: VPUMI40XX.NNDMA
    // CHECK-SAME: port = 0
    // CHECK-SAME: inputs(%[[INPUT]] : memref<[[INPUT_TYPE]]>)
    // CHECK-SAME: outputs(%[[OUTPUT]] : [[OUTPUT_TYPE]])
    // CHECK-SAME: waits(%[[BAR1]] : !VPURegMapped.Index<0:0:1>)
    // CHECK-SAME: start_after(0)
    // CHECK-SAME: clean_after(0)
    // CHECK-SAME: acceleration_mode(<DISABLE>)
    // CHECK-SAME: dma_transaction(#VPUMI40XX.NNDMATransaction<
    // CHECK-SAME:    inputType = memref<[[INPUT_TYPE]]>
    // CHECK-SAME:    outputType = [[OUTPUT_TYPE]]
    // CHECK-SAME: >)
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:1>

    return
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
IE.CNNNetwork entryPoint : @doubleDMAWithDistributedBuffer inputsInfo : {
  DataInfo "dummy_input" : tensor<1x50x1x1xf16>
} outputsInfo : {
  DataInfo "dummy_output" : tensor<1x50x1x1xf16>
}
func.func @doubleDMAWithDistributedBuffer() {
    %cst_3 = const.Declare memref<1x1x1x2688xf16> = dense<0> : tensor<48x1x1x4xsi32>, [#const.FuseWeights<tensor<1x1x1x2688xf16>, weightsTable = <dense<0> : tensor<48x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[768], sparsityPtr=3840 : i64, offsets=[0], weightsTableSize=768 : i64, weightsElemBitSize=16 : i64, weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>, channelOffset=0 : i64, originalOC=0 : i64>]>, weights = <dense<0.0> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]>, sparsity = <dense<0.0> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]>>]
    // CHECK: %[[INPUT:.+]] = const.Declare memref<[[INPUT_TYPE:.+]]> =

    %65 = VPURT.DeclareBuffer <CMX_NN> [0, 1, 2, 3] <0> -> !VPUIP.DistributedBuffer<1x1x1x2688xf16, {order = #NCHW, strides = [2688, 2688, 2688, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    // CHECK-NOT: VPURT.DeclareBuffer .+ -> !VPUIP.DistributedBuffer

    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK-NOT: VPURT.ConfigureBarrier
    // CHECK: %[[BAR0:.+]] = VPUMI40XX.ConfigureBarrier
    // CHECK-SAME: consumer_count = 0
    // CHECK-SAME: producer_count = 2
    // CHECK-SAME: <0, -1>
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    // CHECK: %[[OUTPUT_0:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [0]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_0:.+]]

    // CHECK: %[[OUTPUT_1:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [1]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_1:.+]]

    // CHECK: %[[OUTPUT_2:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [2]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_2:.+]]

    // CHECK: %[[OUTPUT_3:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [3]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_3:.+]]

    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %364 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%cst_3 : memref<1x1x1x2688xf16>) outputs(%65 : !VPUIP.DistributedBuffer<1x1x1x2688xf16, {order = #NCHW, strides = [2688, 2688, 2688, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x1x1x2688xf16, {order = #NCHW, strides = [2688, 2688, 2688, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    }

    // CHECK-NOT: VPURT.Task
    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: VPUMI40XX.NNDMA
    // CHECK-SAME: port = 0
    // CHECK-SAME: inputs(%[[INPUT]] : memref<[[INPUT_TYPE]]>)
    // CHECK-SAME: outputs(%[[OUTPUT_0]], %[[OUTPUT_1]], %[[OUTPUT_2]], %[[OUTPUT_3]] : [[OUTPUT_TYPE_0]], [[OUTPUT_TYPE_1]], [[OUTPUT_TYPE_2]], [[OUTPUT_TYPE_3]])
    // CHECK-SAME: updates(%[[BAR0]] : !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: start_after(0)
    // CHECK-SAME: clean_after(0)
    // CHECK-SAME: acceleration_mode(<DISABLE>)
    // CHECK-SAME: dma_transaction(#VPUMI40XX.NNDMATransaction<
    // CHECK-SAME:    inputType = memref<[[INPUT_TYPE]]>
    // CHECK-SAME:    outputType = !VPUIP.DistributedBuffer
    // CHECK-SAME: >)
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:0>

    // CHECK: %[[OUTPUT1_0:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [0]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_0]]

    // CHECK: %[[OUTPUT1_1:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [1]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_1]]

    // CHECK: %[[OUTPUT1_2:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [2]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_2]]

    // CHECK: %[[OUTPUT1_3:.+]] = VPURT.DeclareBuffer
    // CHECK-SAME: [3]
    // CHECK-NOT: -> !VPUIP.DistributedBuffer
    // CHECK-SAME: -> [[OUTPUT_TYPE_3]]

    VPURT.Task updates(%0 : !VPURT.Barrier) {
      %364 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%cst_3 : memref<1x1x1x2688xf16>) outputs(%65 : !VPUIP.DistributedBuffer<1x1x1x2688xf16, {order = #NCHW, strides = [2688, 2688, 2688, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>) -> !VPUIP.DistributedBuffer<1x1x1x2688xf16, {order = #NCHW, strides = [2688, 2688, 2688, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688], [1, 1, 1, 2688]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>
    }

    // CHECK-NOT: VPURT.Task
    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: VPUMI40XX.NNDMA
    // CHECK-SAME: port = 0
    // CHECK-SAME: inputs(%[[INPUT]] : memref<[[INPUT_TYPE]]>)
    // CHECK-SAME: outputs(%[[OUTPUT1_0]], %[[OUTPUT1_1]], %[[OUTPUT1_2]], %[[OUTPUT1_3]] : [[OUTPUT_TYPE_0]], [[OUTPUT_TYPE_1]], [[OUTPUT_TYPE_2]], [[OUTPUT_TYPE_3]])
    // CHECK-SAME: updates(%[[BAR0]] : !VPURegMapped.Index<0:0:0>)
    // CHECK-SAME: start_after(0)
    // CHECK-SAME: clean_after(0)
    // CHECK-SAME: acceleration_mode(<DISABLE>)
    // CHECK-SAME: dma_transaction(#VPUMI40XX.NNDMATransaction<
    // CHECK-SAME:    inputType = memref<[[INPUT_TYPE]]>
    // CHECK-SAME:    outputType = !VPUIP.DistributedBuffer
    // CHECK-SAME: >)
    // CHECK-SAME: -> !VPURegMapped.Index<0:0:1>

    return
}
