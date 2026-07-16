//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-copies %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>

!OutDataBufferType = !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!OutSMBufferType = !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// CHECK-LABEL: @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUserSparsity
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.+]]: memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>
func.func @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUserSparsity(
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType)
    -> (!OutDataBufferType, !OutSMBufferType, memref<1x128x28x7xf16, {order = #NHWC}, @DDR>, memref<1x128x28x7xi1, {order = #NHWC}, @DDR>)
{
    // alloc for Conv data out
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // alloc for Conv sparsity map out
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output_sparsity_map(%1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        output_sparsity_map(%1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    ->  !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
        !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    // Input 2: Allocated buffer for grouped op output
    %4 = VPURT.AllocDistributed -> !OutDataBufferType
    %5 = VPURT.AllocDistributed -> !OutSMBufferType

    %data7 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 14, 14] : !OutDataBufferType
        to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm7 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 128, 14, 14] : !OutSMBufferType
        to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CMX->CMX copy with two distributed operands
    %data8 = VPUIP.Copy
        inputs(%2#0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%data7 : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm8 = VPUIP.Copy
        inputs(%2#1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm7 : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // alloc for Conv data out
    %10 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // alloc for Conv sparsity map out
    %11 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // Input 1: Convolution
    %12:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%10 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        parent_output_sparsity_map(%11 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%10 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        output_sparsity_map(%11 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    ->  !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>,
        !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> variants : {
        DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %data14 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 14, 14] : !OutDataBufferType
        to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm14 = VPUIP.SubView %5 [0, 128, 0, 0] [1, 128, 14, 14] : !OutSMBufferType
        to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CMX->CMX copy with two distributed operands
    %data15 = VPUIP.Copy
        inputs(%12#0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%data14 : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %sm15 = VPUIP.Copy
        inputs(%12#1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%sm14 : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)  ->  !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %data16 = VPUIP.ConcatView
        inputs(%data8, %data15 : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%4 : !OutDataBufferType) -> !OutDataBufferType
    %sm16 = VPUIP.ConcatView
        inputs(%sm8, %sm15 : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>, !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%5 : !OutSMBufferType) -> !OutSMBufferType

    %17 = memref.alloc() : memref<1x128x28x7xf16, {order = #NHWC}, @DDR>
    %18 = memref.alloc() : memref<1x128x28x7xi1, {order = #NHWC}, @DDR>

    %data20 = VPUIP.GenericReshape
        inputs(%12#0 : !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        -> !VPUIP.DistributedBuffer<1x128x28x7xf16, #NHWC, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    %sm20 = VPUIP.GenericReshape
        inputs(%12#1 : !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        -> !VPUIP.DistributedBuffer<1x128x28x7xi1, #NHWC, @CMX_NN,
        {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    %data21 = VPUIP.Copy
        inputs(%data20 : !VPUIP.DistributedBuffer<1x128x28x7xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%17 : memref<1x128x28x7xf16, {order = #NHWC}, @DDR>)  ->  memref<1x128x28x7xf16, {order = #NHWC}, @DDR>
    %sm21 = VPUIP.Copy
        inputs(%sm20 : !VPUIP.DistributedBuffer<1x128x28x7xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%18 : memref<1x128x28x7xi1, {order = #NHWC}, @DDR>)  ->  memref<1x128x28x7xi1, {order = #NHWC}, @DDR>

    return %data16, %sm16, %data21, %sm21 : !OutDataBufferType, !OutSMBufferType, memref<1x128x28x7xf16, {order = #NHWC}, @DDR>, memref<1x128x28x7xi1, {order = #NHWC}, @DDR>

    // CHECK:      [[BUFF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SUBVIEW_0_DATA:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[NCE_0:%.+]]:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME:     parent_output([[SUBVIEW_0_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     parent_output_sparsity_map([[SUBVIEW_0_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     output_sparsity_map([[SUBVIEW_0_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK:      [[SUBVIEW_1_DATA:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SUBVIEW_1_SM:%.+]] = VPUIP.SubView [[BUFF_1]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:      [[NCE_1:%.+]]:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME:     parent_output([[SUBVIEW_1_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     parent_output_sparsity_map([[SUBVIEW_1_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     output_sparsity_map([[SUBVIEW_1_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK:       [[BUFF_5:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[COPY_TO_CMX_SM:%.+]] = VPUIP.Copy inputs([[NCE_1]]#1
    // CHECK-SAME:  outputs([[BUFF_5]]
    // CHECK:       [[BUFF_4:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[COPY_TO_CMX_DATA:%.+]] = VPUIP.Copy inputs([[NCE_1]]#0
    // CHECK-SAME:  outputs([[BUFF_4]]
    // CHECK:      [[CONCAT_DATA:%.+]] =  VPUIP.ConcatView inputs([[NCE_0]]#0, [[NCE_1]]#0
    // CHECK-SAME:  outputs([[BUFF_0]]
    // CHECK:      [[CONCAT_SM:%.+]] =  VPUIP.ConcatView inputs([[NCE_0]]#1, [[NCE_1]]#1
    // CHECK-SAME:  outputs([[BUFF_1]]

    // CHECK:      [[BUFF_6:%.+]]  = memref.alloc() : memref<1x128x28x7xf16, {order = #NHWC}, @DDR>
    // CHECK:      [[BUFF_7:%.+]] = memref.alloc() : memref<1x128x28x7xi1, {order = #NHWC}, @DDR>
    // CHECK:      [[RESHAPE_DATA:%.+]] = VPUIP.GenericReshape inputs([[COPY_TO_CMX_DATA]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x128x28x7xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[RESHAPE_SM:%.+]] = VPUIP.GenericReshape inputs([[COPY_TO_CMX_SM]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x128x28x7xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[COPY_OUT_DATA:%.+]] = VPUIP.Copy inputs([[RESHAPE_DATA]]
    // CHECK-SAME:  outputs([[BUFF_6]]
    // CHECK:      [[COPY_OUT_SM:%.+]] = VPUIP.Copy inputs([[RESHAPE_SM]]
    // CHECK-SAME:  outputs([[BUFF_7]]

    // CHECK:       return [[CONCAT_DATA]], [[CONCAT_SM]], [[COPY_OUT_DATA]], [[COPY_OUT_SM]]

}
