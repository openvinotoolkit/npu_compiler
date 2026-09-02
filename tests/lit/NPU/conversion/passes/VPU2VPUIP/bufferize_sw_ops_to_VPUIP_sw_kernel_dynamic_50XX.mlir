//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!InputDistributed1 = !VPU.DistributedTensor<1x2x35x512xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 35, 512], [1, 1, 35, 512]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]},
    dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]> : tensor<4xsi64>>

!InputDistributed2 = !VPU.DistributedTensor<1x2x1x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], compute_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]],
    memory_shapes = [[1, 1, 1, 128], [1, 1, 1, 128]], memory_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]}>

!InputDistributed3 = !VPU.DistributedTensor<2x4x128x128xf16, #NCHW, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [2, 1, 1, 1], num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], compute_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]],
    memory_shapes = [[1, 4, 128, 128], [1, 4, 128, 128]], memory_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]}>

!InputDistributed4 = !VPU.DistributedTensor<1x1x1x2496xsi32, #NCHW, @CMX_NN, {
    mode = "DUPLICATED", num_clusters = 2 : i64, uniform_distributed_segments,
    compute_shapes = [[1, 1, 1, 2496], [1, 1, 1, 2496]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 1, 1, 2496], [1, 1, 1, 2496]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]}>

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


// CHECK: func.func nested @builtin_LSTMSequence(memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xsi32, @CMX_NN>, memref<*xf16, @CMX_NN>, memref<*xf16, @CMX_NN>, none, i64, none)
// CHECK-SAME: attributes {VPU.kernel_code = "lstm_dpu.cpp", VPU.kernel_entry = "lstm_dpu", VPU.kernel_name = "lstm_dpu", VPU.task_type = @COMPUTE}
// CHECK-LABEL:  func.func @DynamicLSTMSequence
module attributes {config.platform = #config.platform<NPU5010>} {
func.func @DynamicLSTMSequence(
        %arg1: tensor<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>,
        %arg2: tensor<1x2x1x128xf16>, %arg3: tensor<1x2x1x128xf16>,
        %arg4: tensor<2x4x128x128xf16>, %arg5: tensor<1x1x1x2496xsi32>)
    -> (tensor<1x2x?x128xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 128]> : tensor<4xsi64>, order = #NCHW}>,
        tensor<1x2x1x128xf16>,
        tensor<1x2x1x128xf16>) {
      // CHECK: [[INPUT_DDR:%.+]]: memref<1x2x?x512xf16, {bounds = #const.OpaqueI64Elements<[1, 2, 35, 512]> : tensor<4xsi64>, order = #NCHW}>

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
      %cmx_input4 = VPU.Copy(%arg4) {out_mem_space = @CMX_NN} : tensor<2x4x128x128xf16> -> !InputDistributed3
      %cmx_input5 = VPU.Copy(%arg5) {out_mem_space = @CMX_NN} : tensor<1x1x1x2496xsi32> -> !InputDistributed4

      %outputHiddenValues, %outputHiddenState, %outputCellState = VPU.LSTMSequence(%cmx_input1, %cmx_input2,
        %cmx_input3, %cmx_input4, %cmx_input5)
            {direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0, 1>, useDpu = true}
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
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#C = affine_map<(d0) -> (d0)>

// CHECK: func.func nested @builtin_InterpolateDMA(
// CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
// CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
// CHECK-SAME: memref<*xui8, [@CMX_NN, 0]>
// CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
// CHECK-SAME: ) attributes {VPU.kernel_code = "interpolate_dma.cpp", VPU.kernel_entry = "interpolate_dma", VPU.kernel_name = "interpolate_dma", VPU.task_type = @COMPUTE}

// CHECK-LABEL: func.func @ScaleParameterInterpolateLayerTest(
module attributes {config.platform = #config.platform<NPU5010>} {
func.func @ScaleParameterInterpolateLayerTest(%interp_input: tensor<1x3x4x6xf16>, %scales: tensor<2xf32>)
      -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
    %cst_aux = const.Declare tensor<1x1x1x1024xui8> = dense<0> : tensor<1x1x1x1024xui8>

    %s1 = VPU.AffineReshape(%scales) {dim_mapping = [[0, 1, 2, 3]], shape_value = [1, 1, 1, 2]} : tensor<2xf32> -> tensor<1x1x1x2xf32>
    %s2 = VPU.Convert(%s1) {dstElemType = f16} : tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf16>
    %s3 = VPU.AffineReshape(%s2) {dim_mapping = [[0], [0], [0], [0]], shape_value = [2]} : tensor<1x1x1x2xf16> -> tensor<2xf16>

    %in = VPU.Copy(%interp_input) {out_mem_space = [@CMX_NN, 0]} : tensor<1x3x4x6xf16> -> tensor<1x3x4x6xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    %in_scales = VPU.Copy(%s3) {out_mem_space = [@CMX_NN, 0]} : tensor<2xf16> -> tensor<2xf16, {mem_space = [@CMX_NN, 0], order = #C}>
    %in_aux = VPU.Copy(%cst_aux) {out_mem_space = [@CMX_NN, 0]} : tensor<1x1x1x1024xui8> -> tensor<1x1x1x1024xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>

    %out_cmx = VPU.InterpolateDMA(%in, %in_scales, %in_aux)
      {attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>,
       axes_attr = [2, 3], bounds_representation = #VPU.bounds_representation<BOUNDS>}
      : tensor<1x3x4x6xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>, tensor<2xf16, {mem_space = [@CMX_NN, 0], order = #C}>, tensor<1x1x1x1024xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>

    %out = VPU.Copy(%out_cmx) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
        -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
    return %out : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
  }

  // CHECK: [[CMX_CST32:%.+]] = arith.constant 32
  // CHECK: [[CMX_CST48:%.+]] = arith.constant 48
  // CHECK: [[CMX_OUT:%.+]] = memref.alloc([[CMX_CST32]], [[CMX_CST48]])
  // CHECK-SAME: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}, [@CMX_NN, 0]>

  // CHECK: [[KERNEL_OUT:%.+]]:2 = VPUIP.SW.Kernel
  // CHECK-SAME: @VPU.SW::@builtin_InterpolateDMA
  // CHECK-SAME: outputs([[CMX_OUT]]

  // CHECK: [[DDR_CST32:%.+]] = arith.constant 32
  // CHECK: [[DDR_CST48:%.+]] = arith.constant 48
  // CHECK: [[DDR_OUT:%.+]] = memref.alloc([[DDR_CST32]], [[DDR_CST48]])
  // CHECK-SAME: memref<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

  // CHECK: [[COPY_OUT:%.+]] = VPUIP.Copy inputs([[KERNEL_OUT]]#0 {{.*}} outputs([[DDR_OUT]]
  // CHECK: return [[COPY_OUT]]
}
