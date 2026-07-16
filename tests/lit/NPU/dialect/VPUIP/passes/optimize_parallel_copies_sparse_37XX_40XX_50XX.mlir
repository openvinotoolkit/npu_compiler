//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-parallel-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IODataDDRType = memref<1x32x28x28xf16, {order = #NHWC}, @DDR>
!IOSMDDRType = memref<1x32x28x28xi1, {order = #NHWC}, @DDR>

!IODataCMXType = memref<1x32x28x28xf16, {order = #NHWC}, @CMX_NN>
!IOSMCMXType = memref<1x32x28x28xi1, {order = #NHWC}, @CMX_NN>

!Weights_CMX = memref<32x32x1x1xf16, {order = #NHWC}, @CMX_NN>

!IODataDistrType = !VPUIP.DistributedBuffer<
  1x32x28x28xf16, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

!IOSMDistrType = !VPUIP.DistributedBuffer<
  1x32x28x28xi1, #NHWC, @CMX_NN, {
  mode = DUPLICATED,
  num_clusters = 4 : i64
}>

// CHECK-LABEL: @OptimizeParallelMulticlusterCopiesSparse
func.func @OptimizeParallelMulticlusterCopiesSparse()
        -> (!IODataDistrType, !IOSMDistrType, !IODataDistrType, !IOSMDistrType) {
    %0 = memref.alloc() : !IODataCMXType
    %1 = memref.alloc() : !IOSMCMXType

    %3 = memref.alloc() : !IODataDDRType
    %4 = memref.alloc() : !IOSMDDRType
    %6 = VPUIP.Copy
        inputs(%0 : !IODataCMXType)
        outputs(%3 : !IODataDDRType) -> !IODataDDRType
    %sm6 = VPUIP.Copy
        inputs(%1 : !IOSMCMXType)
        outputs(%4 : !IOSMDDRType) -> !IOSMDDRType

    %7 = VPURT.AllocDistributed -> !IODataDistrType
    %8 = VPURT.AllocDistributed -> !IOSMDistrType
    %in_data_0 = VPUIP.Copy
        inputs(%6 : !IODataDDRType)
        outputs(%7 : !IODataDistrType) -> !IODataDistrType
    %in_sm_0 = VPUIP.Copy
        inputs(%sm6 : !IOSMDDRType)
        outputs(%8 : !IOSMDistrType) -> !IOSMDistrType

    %out_data_0 = VPURT.AllocDistributed -> !IODataDistrType
    %out_sm_0 = VPURT.AllocDistributed -> !IOSMDistrType

    %14 = memref.alloc() : !Weights_CMX

    %16:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          }>
          input(%in_data_0 : !IODataDistrType)
          input_sparsity_map(%in_sm_0 : !IOSMDistrType)
          weights(%14 : !Weights_CMX)
          parent_input(%in_data_0 : !IODataDistrType)
          parent_input_sparsity_map(%in_sm_0 : !IOSMDistrType)
          parent_output(%out_data_0 : !IODataDistrType)
          parent_output_sparsity_map(%out_sm_0 : !IOSMDistrType)
          outputs(%out_data_0 : !IODataDistrType)
          output_sparsity_map(%out_sm_0 : !IOSMDistrType)
            -> !IODataDistrType, !IOSMDistrType variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
       } PPE :  { }

    %17 = VPURT.AllocDistributed -> !IODataDistrType
    %18 =  VPURT.AllocDistributed -> !IOSMDistrType
    %in_data_1 = VPUIP.Copy
        inputs(%6 : !IODataDDRType)
        outputs(%17 : !IODataDistrType) -> !IODataDistrType
    %in_sm_1 = VPUIP.Copy
        inputs(%sm6 : !IOSMDDRType)
        outputs(%18 : !IOSMDistrType) -> !IOSMDistrType

    %out_data_1 = VPURT.AllocDistributed -> !IODataDistrType
    %out_sm_1 =  VPURT.AllocDistributed -> !IOSMDistrType

      %24:2 = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
        kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        kernel_size = [1, 1],
        kernel_strides = [1, 1],
        task_type = #VPUIP.nce_task_type<CONV>
          }>
          input(%in_data_1 : !IODataDistrType)
          input_sparsity_map(%in_sm_1 : !IOSMDistrType)
          weights(%14 : !Weights_CMX)
          parent_input(%in_data_1 : !IODataDistrType)
          parent_input_sparsity_map(%in_sm_1 : !IOSMDistrType)
          parent_output(%out_data_1 : !IODataDistrType)
          parent_output_sparsity_map(%out_sm_1 : !IOSMDistrType)
          outputs(%out_data_1 : !IODataDistrType)
          output_sparsity_map(%out_sm_1 : !IOSMDistrType)
            -> !IODataDistrType, !IOSMDistrType variants :  {
              DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  { }

    return %16#0, %16#1, %24#0, %24#1: !IODataDistrType, !IOSMDistrType, !IODataDistrType, !IOSMDistrType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x32x28x28xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x32x28x28xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_1_DATA:%.+]] = memref.alloc() : memref<1x32x28x28xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BUFF_1_SM:%.+]] = memref.alloc() : memref<1x32x28x28xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[COMMON_ROOT:%.+]] = VPUIP.Copy inputs([[BUFF_0_DATA]]
    // CHECK-SAME:      outputs([[BUFF_1_DATA]]

    // CHECK:       [[COMMON_ROOT_SM:%.+]] = VPUIP.Copy inputs([[BUFF_0_SM]]
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[BUFF_2_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_2_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_0:%.+]] = VPUIP.Copy inputs([[COMMON_ROOT]]
    // CHECK-SAME:      outputs([[BUFF_2_DATA]]

    // CHECK:       [[SM_0:%.+]] = VPUIP.Copy inputs([[COMMON_ROOT_SM]]
    // CHECK-SAME:      outputs([[BUFF_2_SM]]

    // CHECK:       [[DATA_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE0_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]]
    // CHECK-SAME:      outputs([[DATA_1]]

    // CHECK:       [[DATA_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x32x28x28xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE1_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]]
    // CHECK-SAME:      outputs([[DATA_3]]

    // CHECK: return [[NCE0_U]]#0, [[NCE0_U]]#1, [[NCE1_U]]#0, [[NCE1_U]]#1
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataDDRType = memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
!ISMDDRType = memref<1x144x128x128xi1, {order = #NHWC}, @DDR>

!IDataHalfCMXType = memref<1x144x64x128xf16, {order = #NHWC}, @CMX_NN>
!ISMHalfCMXType = memref<1x144x64x128xi1, {order = #NHWC}, @CMX_NN>

!ODistrDataType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSMType = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

// CHECK-LABEL: @OptimizeParallelSubViewWithDistributedCopiesSparse
// CHECK-SAME: [[ARG_0:%[^:]+]]: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: memref<1x144x128x128xi1, {order = #NHWC}, @DDR>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
func.func @OptimizeParallelSubViewWithDistributedCopiesSparse(
        %input: !IDataDDRType,
        %input_sm: !ISMDDRType,
        %weights: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
         -> (!ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType) {

    %0 = memref.alloc() : !IDataDDRType
    %1 = memref.alloc() : !ISMDDRType

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%0 : !IDataDDRType) -> !IDataDDRType
    %sm3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%1 : !ISMDDRType) -> !ISMDDRType

    %4 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm4 = VPUIP.SubView %sm3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %5 = VPURT.AllocDistributed -> !ODistrDataType
    %6 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_0 = VPUIP.Copy
        inputs(%4 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%5 : !ODistrDataType) -> !ODistrDataType
    %in_sm_0 = VPUIP.Copy
        inputs(%sm4 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%6 : !ODistrSMType) -> !ODistrSMType

    %out_data_0 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_0 = VPURT.AllocDistributed -> !ODistrSMType

    %12:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in_data_0 : !ODistrDataType)
        input_sparsity_map(%in_sm_0 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in_data_0 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_0 : !ODistrSMType)
        parent_output(%out_data_0 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_0 : !ODistrSMType)
        outputs(%out_data_0 : !ODistrDataType)
        output_sparsity_map(%out_sm_0 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    %13 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm13 = VPUIP.SubView %sm3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %14 = VPURT.AllocDistributed -> !ODistrDataType
    %15 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_1 = VPUIP.Copy
        inputs(%13 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%14 : !ODistrDataType) -> !ODistrDataType
    %in_sm_1 = VPUIP.Copy
        inputs(%sm13 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%15 : !ODistrSMType) -> !ODistrSMType

    %out_data_1 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_1 = VPURT.AllocDistributed -> !ODistrSMType

    %21:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in_data_1 : !ODistrDataType)
        input_sparsity_map(%in_sm_1 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in_data_1 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_1 : !ODistrSMType)
        parent_output(%out_data_1 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_1 : !ODistrSMType)
        outputs(%out_data_1 : !ODistrDataType)
        output_sparsity_map(%out_sm_1 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    return %12#0, %12#1, %21#0, %21#1 : !ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x144x128x128xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_DATA]]
    // CHECK-SAME:      -> memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[PERMUTE_SM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_SM]]
    // CHECK-SAME:      -> memref<1x144x128x128xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[PERMUTE_SM]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]]
    // CHECK-SAME:      outputs([[BUFF_1_DATA]]

    // CHECK:       [[SM_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_SM]]
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[DATA_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE0_U:%.+]]:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME:      input([[DATA_0]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_output([[DATA_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK-SAME:      outputs([[DATA_1]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK-NOT:   VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-NOT:   VPUIP.Copy

    // CHECK:       [[DATA_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE1_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_0]]
    // CHECK-SAME:      outputs([[DATA_3]]
    // CHECK:       return [[NCE0_U]]#0, [[NCE0_U]]#1, [[NCE1_U]]#0, [[NCE1_U]]#1
}
// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!IDataDDRType = memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
!ISMDDRType = memref<1x144x128x128xi1, {order = #NHWC}, @DDR>

!IDataHalfCMXType = memref<1x144x64x128xf16, {order = #NHWC}, @CMX_NN>
!ISMHalfCMXType = memref<1x144x64x128xi1, {order = #NHWC}, @CMX_NN>

!ODistrDataType = !VPUIP.DistributedBuffer<
    1x144x64x128xf16, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>
!ODistrSMType = !VPUIP.DistributedBuffer<
    1x144x64x128xi1, #NHWC, @CMX_NN, {
    mode = DUPLICATED,
    num_clusters = 4 : i64
}>

// CHECK-LABEL: @NotOptimizeParallelDistributedCopiesWithSubviewHasDiffOffsetSparse
// CHECK-SAME:      [[ARG_0:%[^:]+]]: memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
// CHECK-SAME:      [[ARG_1:%[^:]+]]: memref<1x144x128x128xi1, {order = #NHWC}, @DDR>
// CHECK-SAME:      [[ARG_2:%[^:]+]]: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>
func.func @NotOptimizeParallelDistributedCopiesWithSubviewHasDiffOffsetSparse(
        %input: !IDataDDRType,
        %input_sm: !ISMDDRType,
        %weights: memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
         -> (!ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType) {

    %0 = memref.alloc() : !IDataDDRType
    %1 = memref.alloc() : !ISMDDRType

    %3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%0 : !IDataDDRType) -> !IDataDDRType
    %sm3 = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
            inputs(%1 : !ISMDDRType) -> !ISMDDRType

    %4 = VPUIP.SubView %3 [0, 0, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm4 = VPUIP.SubView %sm3 [0, 0, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %5 = VPURT.AllocDistributed -> !ODistrDataType
    %6 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_0 = VPUIP.Copy
        inputs(%4 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%5 : !ODistrDataType) -> !ODistrDataType
    %in_sm_0 = VPUIP.Copy
        inputs(%sm4 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%6 : !ODistrSMType) -> !ODistrSMType

    %out_data_0 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_0 = VPURT.AllocDistributed -> !ODistrSMType

    %12:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in_data_0 : !ODistrDataType)
        input_sparsity_map(%in_sm_0 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in_data_0 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_0 : !ODistrSMType)
        parent_output(%out_data_0 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_0 : !ODistrSMType)
        outputs(%out_data_0 : !ODistrDataType)
        output_sparsity_map(%out_sm_0 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    %13 = VPUIP.SubView %3 [0, 1, 64, 0] [1, 144, 64, 128] : !IDataDDRType
        to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    %sm13 = VPUIP.SubView %sm3 [0, 1, 64, 0] [1, 144, 64, 128] : !ISMDDRType
        to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    %14 = VPURT.AllocDistributed -> !ODistrDataType
    %15 = VPURT.AllocDistributed -> !ODistrSMType
    %in_data_1 = VPUIP.Copy
        inputs(%13 : memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%14 : !ODistrDataType) -> !ODistrDataType
    %in_sm_1 = VPUIP.Copy
        inputs(%sm13 : memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>)
        outputs(%15 : !ODistrSMType) -> !ODistrSMType

    %out_data_1 = VPURT.AllocDistributed -> !ODistrDataType
    %out_sm_1 = VPURT.AllocDistributed -> !ODistrSMType

    %21:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%in_data_1 : !ODistrDataType)
        input_sparsity_map(%in_sm_1 : !ODistrSMType)
        weights(%weights : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%in_data_1 : !ODistrDataType)
        parent_input_sparsity_map(%in_sm_1 : !ODistrSMType)
        parent_output(%out_data_1 : !ODistrDataType)
        parent_output_sparsity_map(%out_sm_1 : !ODistrSMType)
        outputs(%out_data_1 : !ODistrDataType)
        output_sparsity_map(%out_sm_1 : !ODistrSMType)
            -> !ODistrDataType , !ODistrSMType variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [15, 5, 31], mpe_mode = #VPU.mpe_mode<VECTOR_FP16>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
    }

    return %12#0, %12#1, %21#0, %21#1 : !ODistrDataType, !ODistrSMType, !ODistrDataType, !ODistrSMType

    // CHECK:       [[BUFF_0_DATA:%.+]] = memref.alloc() : memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[BUFF_0_SM:%.+]] = memref.alloc() : memref<1x144x128x128xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[PERMUTE:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_DATA]]
    // CHECK-SAME:      -> memref<1x144x128x128xf16, {order = #NHWC}, @DDR>
    // CHECK:       [[PERMUTE_SM:%.+]] = VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      inputs([[BUFF_0_SM]]
    // CHECK-SAME:      -> memref<1x144x128x128xi1, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SUBVIEW_0_SM:%.+]] = VPUIP.SubView [[PERMUTE_SM]] [0, 0, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFF_1_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_1_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0]]
    // CHECK-SAME:      outputs([[BUFF_1_DATA]]
    // CHECK:       [[SM_0:%.+]] = VPUIP.Copy inputs([[SUBVIEW_0_SM]]
    // CHECK-SAME:      outputs([[BUFF_1_SM]]

    // CHECK:       [[DATA_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE0_U:%.+]]:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 9240 : i64} <{kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, kernel_size = [1, 1], kernel_strides = [1, 1],
    // CHECK-SAME:  task_type = #VPUIP.nce_task_type<CONV>}>
    // CHECK-SAME:      input([[DATA_0]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      input_sparsity_map([[SM_0]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_input_sparsity_map([[SM_0]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      output([[DATA_1]]
    // CHECK-SAME:      output_sparsity_map([[SM_1]]

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[PERMUTE]] [0, 1, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xf16, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>
    // CHECK:       [[SUBVIEW_1_SM:%.+]] = VPUIP.SubView [[PERMUTE_SM]] [0, 1, 64, 0] [1, 144, 64, 128]
    // CHECK-SAME:      to memref<1x144x64x128xi1, {order = #NHWC, strides = [2359296, 1, 18432, 144]}, @DDR>

    // CHECK:       [[BUFF_3_DATA:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_3_SM:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[DATA_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1]]
    // CHECK-SAME:      outputs([[BUFF_3_DATA]]
    // CHECK:       [[SM_2:%.+]] = VPUIP.Copy inputs([[SUBVIEW_1_SM]]
    // CHECK-SAME:      outputs([[BUFF_3_SM]]

    // CHECK:       [[DATA_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>
    // CHECK:       [[SM_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>

    // CHECK:       [[NCE1_U:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[DATA_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      input_sparsity_map([[SM_2]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      weights([[ARG_2]] : memref<32x144x1x1xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_input([[DATA_2]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_input_sparsity_map([[SM_2]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_output([[DATA_3]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      parent_output_sparsity_map([[SM_3]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      outputs([[DATA_3]] : !VPUIP.DistributedBuffer<1x144x64x128xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)
    // CHECK-SAME:      output_sparsity_map([[SM_3]] : !VPUIP.DistributedBuffer<1x144x64x128xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64}>)

    // CHECK:       return [[NCE0_U]]#0, [[NCE0_U]]#1, [[NCE1_U]]#0, [[NCE1_U]]#1
}
