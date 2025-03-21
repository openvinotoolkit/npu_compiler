//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --uniquify-weights-table-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutDistType = !VPUIP.DistributedBuffer<
    1x1152x1x1xf16, #NHWC, @CMX_NN,
    {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 4, 1, 1], num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1], [1, 288, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 288, 0, 0], [0, 576, 0, 0], [0, 864, 0, 0]],
    memory_shapes = [[1, 1152, 1, 1], [1, 1152, 1, 1], [1, 1152, 1, 1], [1, 1152, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!InDistType = !VPUIP.DistributedBuffer<
    1x128x1x1xf16, #NHWC, @CMX_NN,
    {mode = "DUPLICATED", num_clusters = 4 : i64, alignment = [1, 16, 1, 1], uniform_distributed_segments,
    compute_shapes = [[1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1], [1, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

!WeightsDistType = !VPUIP.DistributedBuffer<
    1152x128x1x1xf16, #NHWC, @CMX_NN,
    {mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
    compute_shapes = [[288, 128, 1, 1], [288, 128, 1, 1], [288, 128, 1, 1], [288, 128, 1, 1]],
    compute_offsets = [[0, 0, 0, 0], [288, 0, 0, 0], [576, 0, 0, 0], [864, 0, 0, 0]],
    memory_shapes = [[288, 128, 1, 1], [288, 128, 1, 1], [288, 128, 1, 1], [288, 128, 1, 1]],
    memory_offsets = [[0, 0, 0, 0], [288, 0, 0, 0], [576, 0, 0, 0], [864, 0, 0, 0]]}>

!WeightTableDistType = !VPUIP.DistributedBuffer<
    1152x1x1x4xsi32, #NCHW, @CMX_NN, {
        mode = "SEGMENTED", num_tiles = [4, 1, 1, 1], num_clusters = 4 : i64, alignment = [16, 1, 1, 1], uniform_distributed_segments,
        compute_shapes = [[288, 1, 1, 4], [288, 1, 1, 4], [288, 1, 1, 4], [288, 1, 1, 4]],
        compute_offsets = [[0, 0, 0, 0], [288, 0, 0, 0], [576, 0, 0, 0], [864, 0, 0, 0]],
        memory_shapes = [[288, 1, 1, 4], [288, 1, 1, 4], [288, 1, 1, 4], [288, 1, 1, 4]],
        memory_offsets = [[0, 0, 0, 0], [288, 0, 0, 0], [576, 0, 0, 0], [864, 0, 0, 0]]}>

// CHECK-LABEL: func.func @UniquifyWeightsTableCopies
// CHECK-SAME:      ([[IN1:%.+]]: memref<1x128x1x1xf16, #NHWC>, [[IN2:%.+]]: memref<1x128x1x1xf16, #NHWC>,
// CHECK-SAME:      [[WEIGHTS1:%.+]]: memref<1152x128x1x1xf16, #NHWC>, [[WEIGHTS2:%.+]]: memref<1152x128x1x1xf16, #NHWC>)
func.func @UniquifyWeightsTableCopies(%input1: memref<1x128x1x1xf16, #NHWC>, %input2: memref<1x128x1x1xf16, #NHWC>,
                            %weights1: memref<1152x128x1x1xf16, #NHWC>, %weights2: memref<1152x128x1x1xf16, #NHWC>) -> !OutDistType {
    
    %wt = const.Declare memref<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>
    %in1_alloc = VPURT.AllocDistributed -> !InDistType
    %in1_cmx = VPUIP.Copy inputs(%input1 : memref<1x128x1x1xf16, #NHWC>) outputs(%in1_alloc : !InDistType) -> !InDistType
    %weights1_alloc = VPURT.AllocDistributed -> !WeightsDistType
    %weights1_cmx = VPUIP.Copy inputs(%weights1 : memref<1152x128x1x1xf16, #NHWC>) outputs(%weights1_alloc : !WeightsDistType) -> !WeightsDistType
    %wt1_alloc = VPURT.AllocDistributed -> !WeightTableDistType
    %wt1_cmx = VPUIP.Copy inputs(%wt : memref<1152x1x1x4xsi32>) outputs(%wt1_alloc : !WeightTableDistType) -> !WeightTableDistType
    %conv1_out = VPURT.AllocDistributed -> !OutDistType
    %conv1 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                  kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1029 : i64,
                                  mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
            input(%in1_cmx : !InDistType) weights(%weights1_cmx : !WeightsDistType)
            weight_table(%wt1_cmx : !WeightTableDistType) parent_input(%in1_cmx : !InDistType)
            parent_output(%conv1_out : !OutDistType) outputs(%conv1_out : !OutDistType) -> !OutDistType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 287], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 575], outStart = [0, 0, 288],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 863], outStart = [0, 0, 576],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1151], outStart = [0, 0, 864],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
    %in2_alloc = VPURT.AllocDistributed -> !InDistType
    %in2_cmx = VPUIP.Copy inputs(%input2 : memref<1x128x1x1xf16, #NHWC>) outputs(%in2_alloc : !InDistType) -> !InDistType
    %weights2_alloc = VPURT.AllocDistributed -> !WeightsDistType
    %weights2_cmx = VPUIP.Copy inputs(%weights2 : memref<1152x128x1x1xf16, #NHWC>) outputs(%weights2_alloc : !WeightsDistType) -> !WeightsDistType
    %wt2_alloc = VPURT.AllocDistributed -> !WeightTableDistType
    %wt2_cmx = VPUIP.Copy inputs(%wt : memref<1152x1x1x4xsi32>) outputs(%wt2_alloc : !WeightTableDistType) -> !WeightTableDistType
    %conv2_out = VPURT.AllocDistributed -> !OutDistType
    %conv2 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                  kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1029 : i64,
                                  mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
            input(%in2_cmx : !InDistType) weights(%weights2_cmx : !WeightsDistType)
            weight_table(%wt2_cmx : !WeightTableDistType) parent_input(%in2_cmx : !InDistType)
            parent_output(%conv2_out : !OutDistType) outputs(%conv2_out : !OutDistType) -> !OutDistType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 287], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 1 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 575], outStart = [0, 0, 288],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 2 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 863], outStart = [0, 0, 576],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
      DPUTask {cluster_id = 3 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1151], outStart = [0, 0, 864],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
    return %conv2 : !OutDistType

    // CHECK:     [[WT:%.+]] = const.Declare memref<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>

    // CHECK:     [[IN1_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[IN1_CMX:%.+]] = VPUIP.Copy inputs([[IN1]] : memref<1x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[IN1_ALLOC]] : !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN

    // CHECK:     [[WEIGHTS1_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[WEIGHTS1_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS1]] : memref<1152x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[WEIGHTS1_ALLOC]] : !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN

    // CHECK:     [[WT_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1152x1x1x4xsi32, #NCHW, @CMX_NN
    // CHECK:     [[WT_CMX:%.+]] = VPUIP.Copy inputs([[WT]] : memref<1152x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[WT_ALLOC]] : !VPUIP.DistributedBuffer<1152x1x1x4xsi32, #NCHW, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1152x1x1x4xsi32, #NCHW, @CMX_NN

    // CHECK:     [[CONV1_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[CONV1:%.+]] = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      input([[IN1_CMX]] : !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      weights([[WEIGHTS1_CMX]] : !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      weight_table([[WT_CMX]] : !VPUIP.DistributedBuffer<1152x1x1x4xsi32, #NCHW, @CMX_NN
    // CHECK-SAME:      parent_input([[IN1_CMX]] : !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      parent_output([[CONV1_OUT]] : !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      outputs([[CONV1_OUT]] : !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN

    // CHECK:     [[IN2_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[IN2_CMX:%.+]] = VPUIP.Copy inputs([[IN2]] : memref<1x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[IN2_ALLOC]] : !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[WEIGHTS2_ALLOC:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[WEIGHTS2_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS2]] : memref<1152x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[WEIGHTS2_ALLOC]] : !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[CONV2_OUT:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
    // CHECK:     [[CONV2:%.+]] = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      input([[IN2_CMX]] : !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      weights([[WEIGHTS2_CMX]] : !VPUIP.DistributedBuffer<1152x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      weight_table([[WT_CMX]] : !VPUIP.DistributedBuffer<1152x1x1x4xsi32, #NCHW, @CMX_NN
    // CHECK-SAME:      parent_input([[IN2_CMX]] : !VPUIP.DistributedBuffer<1x128x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      parent_output([[CONV2_OUT]] : !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      outputs([[CONV2_OUT]] : !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
    // CHECK-SAME:      -> !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN

    // CHECK:     return [[CONV2]] : !VPUIP.DistributedBuffer<1x1152x1x1xf16, #NHWC, @CMX_NN
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!OutDistType = memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
!InDistType = memref<1x128x1x1xf16, #NHWC, @CMX_NN>
!WeightsDistType = memref<1152x128x1x1xf16, #NHWC, @CMX_NN>
!WeightTableDistType = memref<1152x1x1x4xsi32, #NCHW, @CMX_NN>

// CHECK-LABEL: func.func @DoNotUniquifyWhenSiblingViewOp
// CHECK-SAME:      ([[IN1:%.+]]: memref<1x128x1x1xf16, #NHWC>, [[IN2:%.+]]: memref<1x128x1x1xf16, #NHWC>,
// CHECK-SAME:      [[WEIGHTS1:%.+]]: memref<1152x128x1x1xf16, #NHWC>, [[WEIGHTS2:%.+]]: memref<1152x128x1x1xf16, #NHWC>)
func.func @DoNotUniquifyWhenSiblingViewOp(%input1: memref<1x128x1x1xf16, #NHWC>, %input2: memref<1x128x1x1xf16, #NHWC>,
                            %weights1: memref<1152x128x1x1xf16, #NHWC>, %weights2: memref<1152x128x1x1xf16, #NHWC>) -> !OutDistType {
    
    %wt = const.Declare memref<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>
    %in1_alloc = memref.alloc() : !InDistType
    %in1_cmx = VPUIP.Copy inputs(%input1 : memref<1x128x1x1xf16, #NHWC>) outputs(%in1_alloc : !InDistType) -> !InDistType
    %weights1_alloc = memref.alloc() : !WeightsDistType
    %weights1_cmx = VPUIP.Copy inputs(%weights1 : memref<1152x128x1x1xf16, #NHWC>) outputs(%weights1_alloc : !WeightsDistType) -> !WeightsDistType
    %wt1_alloc = memref.alloc() : !WeightTableDistType
    %wt1_cmx = VPUIP.Copy inputs(%wt : memref<1152x1x1x4xsi32>) outputs(%wt1_alloc : !WeightTableDistType) -> !WeightTableDistType
    %conv1_out = memref.alloc() : !OutDistType
    %conv1 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                  kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1029 : i64,
                                  mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
            input(%in1_cmx : !InDistType) weights(%weights1_cmx : !WeightsDistType)
            weight_table(%wt1_cmx : !WeightTableDistType) parent_input(%in1_cmx : !InDistType)
            parent_output(%conv1_out : !OutDistType) outputs(%conv1_out : !OutDistType) -> !OutDistType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1151], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
    %in2_alloc = memref.alloc() : !InDistType
    %in2_cmx = VPUIP.Copy inputs(%input2 : memref<1x128x1x1xf16, #NHWC>) outputs(%in2_alloc : !InDistType) -> !InDistType
    %weights2_alloc = memref.alloc() : !WeightsDistType
    %weights2_cmx = VPUIP.Copy inputs(%weights2 : memref<1152x128x1x1xf16, #NHWC>) outputs(%weights2_alloc : !WeightsDistType) -> !WeightsDistType
    %re = VPUIP.GenericReshape inputs(%wt :  memref<1152x1x1x4xsi32>) ->  memref<1x1152x1x4xsi32>
    %reshape = VPUIP.GenericReshape inputs(%re :  memref<1x1152x1x4xsi32>) ->  memref<1152x1x1x4xsi32>
    %wt2_alloc = memref.alloc() : !WeightTableDistType
    %wt2_cmx = VPUIP.Copy inputs(%reshape : memref<1152x1x1x4xsi32>) outputs(%wt2_alloc : !WeightTableDistType) -> !WeightTableDistType
    %conv2_out = memref.alloc() : !OutDistType
    %conv2 = VPUIP.NCEClusterTask {is_zero_offset_weights_table, kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                  kernel_size = [1, 1], kernel_strides = [1, 1], minimumHardwareExecutionCost = 1029 : i64,
                                  mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, task_type = #VPUIP.nce_task_type<CONV>}
            input(%in2_cmx : !InDistType) weights(%weights2_cmx : !WeightsDistType)
            weight_table(%wt2_cmx : !WeightTableDistType) parent_input(%in2_cmx : !InDistType)
            parent_output(%conv2_out : !OutDistType) outputs(%conv2_out : !OutDistType) -> !OutDistType variants : {
      DPUTask {cluster_id = 0 : i64, inEnd = [0, 0, 127], inStart = [0, 0, 0],
        mpe_mode = #VPU.mpe_mode<CUBOID_4x16>, outEnd = [0, 0, 1151], outStart = [0, 0, 0],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
      PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
              lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
    }
    return %conv2 : !OutDistType
    // CHECK:     [[WT:%.+]] = const.Declare memref<1152x1x1x4xsi32> = dense<1> : tensor<1152x1x1x4xsi32>

    // CHECK:     [[IN1_ALLOC:%.+]] = memref.alloc() : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[IN1_CMX:%.+]] = VPUIP.Copy inputs([[IN1]] : memref<1x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[IN1_ALLOC]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x128x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:     [[WEIGHTS1_ALLOC:%.+]] = memref.alloc() : memref<1152x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[WEIGHTS1_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS1]] : memref<1152x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[WEIGHTS1_ALLOC]] : memref<1152x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<1152x128x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:     [[WT_ALLOC:%.+]] = memref.alloc() : memref<1152x1x1x4xsi32, @CMX_NN>
    // CHECK:     [[WT_CMX:%.+]] = VPUIP.Copy inputs([[WT]] : memref<1152x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[WT_ALLOC]] : memref<1152x1x1x4xsi32, @CMX_NN>) -> memref<1152x1x1x4xsi32, @CMX_NN>

    // CHECK:     [[CONV1_OUT:%.+]] = memref.alloc() : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[CONV1:%.+]] = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      input([[IN1_CMX]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      weights([[WEIGHTS1_CMX]] : memref<1152x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      weight_table([[WT_CMX]] : memref<1152x1x1x4xsi32, @CMX_NN>
    // CHECK-SAME:      parent_input([[IN1_CMX]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      parent_output([[CONV1_OUT]] : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[CONV1_OUT]] : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      -> memref<1x1152x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:     [[IN2_ALLOC:%.+]] = memref.alloc() : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[IN2_CMX:%.+]] = VPUIP.Copy inputs([[IN2]] : memref<1x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[IN2_ALLOC]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[WEIGHTS2_ALLOC:%.+]] = memref.alloc() : memref<1152x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[WEIGHTS2_CMX:%.+]] = VPUIP.Copy inputs([[WEIGHTS2]] : memref<1152x128x1x1xf16, #NHWC>)
    // CHECK-SAME:      outputs([[WEIGHTS2_ALLOC]] : memref<1152x128x1x1xf16, #NHWC, @CMX_NN>) -> memref<1152x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[WT_RESHAPE0:%.+]] = VPUIP.GenericReshape inputs([[WT]] : memref<1152x1x1x4xsi32>) -> memref<1x1152x1x4xsi32>
    // CHECK:     [[WT_RESHAPE1:%.+]] = VPUIP.GenericReshape inputs([[WT_RESHAPE0]] : memref<1x1152x1x4xsi32>) -> memref<1152x1x1x4xsi32>
    // CHECK:     [[WT2_ALLOC:%.+]] = memref.alloc() : memref<1152x1x1x4xsi32, @CMX_NN>
    // CHECK:     [[WT2_CMX:%.+]] = VPUIP.Copy inputs([[WT_RESHAPE1]] : memref<1152x1x1x4xsi32>)
    // CHECK-SAME:      outputs([[WT2_ALLOC]] : memref<1152x1x1x4xsi32, @CMX_NN>) -> memref<1152x1x1x4xsi32, @CMX_NN>
    // CHECK:     [[CONV2_OUT:%.+]] = memref.alloc() : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
    // CHECK:     [[CONV2:%.+]] = VPUIP.NCEClusterTask {is_zero_offset_weights_table,
    // CHECK-SAME:      input([[IN2_CMX]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      weights([[WEIGHTS2_CMX]] : memref<1152x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      weight_table([[WT2_CMX]] : memref<1152x1x1x4xsi32, @CMX_NN>
    // CHECK-SAME:      parent_input([[IN2_CMX]] : memref<1x128x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      parent_output([[CONV2_OUT]] : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      outputs([[CONV2_OUT]] : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:      -> memref<1x1152x1x1xf16, #NHWC, @CMX_NN>

    // CHECK:     return [[CONV2]] : memref<1x1152x1x1xf16, #NHWC, @CMX_NN>
}
