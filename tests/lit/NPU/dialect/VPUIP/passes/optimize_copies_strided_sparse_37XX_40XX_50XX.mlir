//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>

!OutDataBufferType = memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>
!OutSMBufferType = memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>

// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeSparsity
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x256x28x28xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x256x28x28xi1, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.+]]: memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>
func.func @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeSparsity(
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType)
    -> (!OutDataBufferType, !OutSMBufferType, memref<1x128x7x28xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x7x28xi1, {order = #NHWC}, @CMX_NN>)
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        parent_output_sparsity_map(%1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        output_sparsity_map(%1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        -> memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    // Input 2: Allocated buffer for grouped op output
    %4 = memref.alloc() : !OutDataBufferType
    %5 = memref.alloc() : !OutSMBufferType

    %data7 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 14, 14] : !OutDataBufferType
        to memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %sm7 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 128, 14, 14] : !OutSMBufferType
        to memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CMX->CMX copy with two distributed operands
    %data8 = VPUIP.Copy inputs(%2#0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%data7 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %sm8 = VPUIP.Copy inputs(%2#1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
            outputs(%sm7 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %data9 = VPUIP.GenericReshape inputs(%2#0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        -> memref<1x128x7x28xf16, {order = #NHWC}, @CMX_NN>

    %sm9 = VPUIP.GenericReshape inputs(%2#1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        -> memref<1x128x7x28xi1, {order = #NHWC}, @CMX_NN>

    // alloc for Conv data out
    %10 = memref.alloc() : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>
    // alloc for Conv sparsity map out
    %11 = memref.alloc() : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>

    // Input 1: Convolution
    %12:2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 34660 : i64, resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%10 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        parent_output_sparsity_map(%11 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        outputs(%10 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
        output_sparsity_map(%11 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
        -> memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>
        variants : {
            DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
            DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [13, 13, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>}
        } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %data14 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 14, 14] : !OutDataBufferType
        to memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %sm14 = VPUIP.SubView %5 [0, 128, 0, 0] [1, 128, 14, 14] : !OutSMBufferType
        to memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    // CMX->CMX copy with two distributed operands
    %data15 = VPUIP.Copy inputs(%12#0 : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>)
            outputs(%data14 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %sm15 = VPUIP.Copy inputs(%12#1 : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>)
            outputs(%sm14 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %data16 = VPUIP.ConcatView inputs(%data8, %data15 :
        memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
        memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    ) outputs(%4 : !OutDataBufferType) -> !OutDataBufferType

    %sm16 = VPUIP.ConcatView inputs(%sm8, %sm15 :
        memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>,
        memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    ) outputs(%5 : !OutSMBufferType) -> !OutSMBufferType

    return %data16, %sm16, %data9, %sm9 :
        !OutDataBufferType, !OutSMBufferType,
        memref<1x128x7x28xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x7x28xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_SM:%.+]] = memref.alloc() : memref<1x256x14x14xi1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUFF_DATA:%.+]] = memref.alloc() : memref<1x256x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[SUBVIEW_DATA:%.+]] = VPUIP.SubView [[BUFF_DATA]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:      to memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    // CHECK:       [[SUBVIEW_SM:%.+]] = VPUIP.SubView [[BUFF_SM]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:      to memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    // CHECK:       [[CLUST_TASK:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      output([[SUBVIEW_DATA]]
    // CHECK-SAME:      output_sparsity_map([[SUBVIEW_SM]]

    // CHECK:       [[BUFF_SM1:%.+]] = memref.alloc() : memref<1x128x14x14xi1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY_OUT_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CLUST_TASK]]#1
    // CHECK-SAME:      outputs([[BUFF_SM1]]
    // CHECK:       [[BUFF_DATA1:%.+]] = memref.alloc() : memref<1x128x14x14xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY_OUT_DATA:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CLUST_TASK]]#0
    // CHECK-SAME:      outputs([[BUFF_DATA1]]

    // CHECK:       [[RESHAPE_DATA:%.+]] = VPUIP.GenericReshape inputs([[COPY_OUT_DATA]]
    // CHECK-SAME:      -> memref<1x128x7x28xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[RESHAPE_SM:%.+]] = VPUIP.GenericReshape inputs([[COPY_OUT_SM]]
    // CHECK-SAME:      -> memref<1x128x7x28xi1, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[SUBVIEW1_DATA:%.+]] = VPUIP.SubView [[BUFF_DATA]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK:       [[SUBVIEW1_SM:%.+]] = VPUIP.SubView [[BUFF_SM]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK:       [[CLUST_TASK1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      output([[SUBVIEW1_DATA]]
    // CHECK-SAME:      output_sparsity_map([[SUBVIEW1_SM]]

    // CHECK:       [[CONCAT_DATA:%.+]] =  VPUIP.ConcatView inputs([[CLUST_TASK]]#0, [[CLUST_TASK1]]#0
    // CHECK:       [[CONCAT_SM:%.+]] =  VPUIP.ConcatView inputs([[CLUST_TASK]]#1, [[CLUST_TASK1]]#1

    // CHECK:       return [[CONCAT_DATA]], [[CONCAT_SM]], [[RESHAPE_DATA]], [[RESHAPE_SM]]
}
