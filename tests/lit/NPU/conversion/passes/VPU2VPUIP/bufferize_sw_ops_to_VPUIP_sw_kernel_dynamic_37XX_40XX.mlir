//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

!InputDistributed1 = !VPU.DistributedTensor<1x2x35x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]},
    dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]> : tensor<4xsi64>>

!InputDistributed2 = !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!InputDistributed3 = !VPU.DistributedTensor<2x4x128x128xf16, #NWHC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
    memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

!InputDistributed4 = !VPU.DistributedTensor<1x1x1x2xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 2], [1, 1, 1, 2]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

!OutputDistributed1 = !VPU.DistributedTensor<1x2x35x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 35, 128], [1, 1, 35, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 35, 128], [1, 1, 35, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]},
    dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]> : tensor<4xsi64>>

!OutputDistributed2 = !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!OutputDistributed3 = !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>


// CHECK-LABEL:  func.func @DynamicLSTMSequence
// CHECK-SAME: [[INPUT_DDR:%.+]]: memref<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
func.func @DynamicLSTMSequence(
        %arg1: tensor<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>,
        %arg2: tensor<1x2x1x128xf16>, %arg3: tensor<1x2x1x128xf16>,
        %arg4: tensor<2x4x128x128xf16, {order = #NWHC}>, %arg5: tensor<1x1x1x2xsi32>)
    -> (tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>,
        tensor<1x2x1x128xf16>,
        tensor<1x2x1x128xf16>) {

      // CHECK: [[IN_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x35x512xf16
      // CHECK: [[IN_SHAPE:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<4xsi32
      // CHECK: [[IN_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[IN_DATA]], [[IN_SHAPE]])

      // CHECK: [[IN_CMX:%.+]] = VPUIP.Copy
      // CHECK-SAME:      inputs([[INPUT_DDR]]
      // CHECK-SAME:      outputs([[IN_BOUNDED_BUFFER]]

      // CHECK: [[OUT_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x2x35x128xf16
      // CHECK: [[OUT_SHAPE:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<4xsi32
      // CHECK: [[OUT_BOUNDED_BUFFER:%.+]] = VPUIP.GroupBoundedBuffer([[OUT_DATA]], [[OUT_SHAPE]])

      %cmx_input1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN}
          : tensor<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>
          -> !InputDistributed1
      %cmx_input2 = VPU.Copy(%arg2) {out_mem_space = @CMX_NN} : tensor<1x2x1x128xf16> -> !InputDistributed2
      %cmx_input3 = VPU.Copy(%arg3) {out_mem_space = @CMX_NN} : tensor<1x2x1x128xf16> -> !InputDistributed2
      %cmx_input4 = VPU.Copy(%arg4) {out_mem_space = @CMX_NN} : tensor<2x4x128x128xf16, {order = #NWHC}> -> !InputDistributed3
      %cmx_input5 = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x1x1x2xsi32> -> !InputDistributed4

      %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%cmx_input1, %cmx_input2,
        %cmx_input3, %cmx_input4, %cmx_input5)
            {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1>}
        : !InputDistributed1, !InputDistributed2, !InputDistributed2, !InputDistributed3, !InputDistributed4
        -> !OutputDistributed1, !OutputDistributed2, !OutputDistributed3

      // CHECK: VPUIP.SW.Kernel
      // CHECK-SAME: resultSegmentSizes
      // CHECK-SAME: @VPU.SW::@builtin_LSTMSequence
      // CHECK-SAME: inputs([[IN_CMX]]
      // CHECK-SAME: outputs([[OUT_BOUNDED_BUFFER]]

      %res1 = VPU.Copy(%outputHiddenValues) : !OutputDistributed1
          -> tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>
      %res2 = VPU.Copy(%outputHiddenState) : !OutputDistributed2 -> tensor<1x2x1x128xf16>
      %res3 = VPU.Copy(%outputCellState) : !OutputDistributed3 -> tensor<1x2x1x128xf16>

      return %res1, %res2, %res3
        : tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>,
          tensor<1x2x1x128xf16>,
          tensor<1x2x1x128xf16>
}
