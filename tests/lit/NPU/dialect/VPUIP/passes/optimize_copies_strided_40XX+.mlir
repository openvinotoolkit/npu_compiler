//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-copies %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
!qElemType1 = !quant.uniform<u8:f16, 6.7832517137714463:123>
!distributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType1 = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType2 = !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!strideDistributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
// CHECK-LABEL: @RemoveCMXToCMXClusteringCopyAndInsertNewCopy
// CHECK-SAME:  [[INPUT1:%.+]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
// CHECK-SAME:  [[INPUT2:%.+]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
func.func @RemoveCMXToCMXClusteringCopyAndInsertNewCopy(%arg0 : !distributeType, %arg1 : !distributeType)
                                    -> (!distributeType1, !distributeType2) {
    %0 = VPURT.AllocDistributed -> !distributeType
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg0 : !distributeType)
        weights(%arg0 : !distributeType)
        parent_input(%arg0 : !distributeType)
        parent_output(%0 : !distributeType)
        outputs(%0 : !distributeType)
    ->  !distributeType variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = VPUIP.QuantizeCast inputs(%1 : !distributeType) -> !distributeType1
    %3 = VPURT.AllocDistributed -> !distributeType1
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%2 : !distributeType1)
        weights(%2 : !distributeType1)
        parent_input(%2 : !distributeType1)
        parent_output(%3 : !distributeType1)
        outputs(%3 : !distributeType1)
    ->  !distributeType1 variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %5 = VPURT.AllocDistributed -> !distributeType
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
    <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg1 : !distributeType)
        weights(%arg1 : !distributeType)
        parent_input(%arg1 : !distributeType)
        parent_output(%5 : !distributeType)
        outputs(%5 : !distributeType)
    ->  !distributeType variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %7 = VPURT.AllocDistributed -> !distributeType2
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : !distributeType2 to !strideDistributeType
    %9 = VPUIP.Copy
        inputs(%1 : !distributeType)
        outputs(%8 : !strideDistributeType)  ->  !strideDistributeType

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : !distributeType2 to !strideDistributeType
    %11 = VPUIP.Copy
        inputs(%6 : !distributeType)
        outputs(%10 : !strideDistributeType)  ->  !strideDistributeType
    %12 = VPUIP.ConcatView
        inputs(%9, %11 : !strideDistributeType, !strideDistributeType)
        outputs(%7 : !distributeType2) -> !distributeType2

    return %4, %12 : !distributeType1, !distributeType2

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]

    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[BUFF_1:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[TILINGCOPY_TO_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[TILINGCOPY_TO_CMX]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUFF_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[ADD_1:%.+]] =  VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     input([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[BUFF_3]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_3]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]

    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     input([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :
    // CHECK:       return  [[ADD_1]], [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
!distributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType1 = !VPUIP.DistributedBuffer<1x32x48x176x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType2 = !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!strideDistributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
// CHECK-LABEL: @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUser
// CHECK-SAME:  [[INPUT1:%.+]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
// CHECK-SAME:  [[INPUT2:%.+]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
func.func @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUser(%arg0 : !distributeType, %arg1 : !distributeType)
                                    -> (memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>, !distributeType2) {
    %0 = VPURT.AllocDistributed -> !distributeType
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
      <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg0 : !distributeType)
        weights(%arg0 : !distributeType)
        parent_input(%arg0 : !distributeType)
        parent_output(%0 : !distributeType)
        outputs(%0 : !distributeType)
    ->  !distributeType variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %2 = VPUIP.GenericReshape inputs(%1 : !distributeType) -> !distributeType1
    %3 = memref.alloc() : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy
        inputs(%2 : !distributeType1)
        outputs(%3 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>

    %5 = VPURT.AllocDistributed -> !distributeType
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg1 : !distributeType)
        weights(%arg1 : !distributeType)
        parent_input(%arg1 : !distributeType)
        parent_output(%5 : !distributeType)
        outputs(%5 : !distributeType)
    ->  !distributeType variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }

    %7 = VPURT.AllocDistributed -> !distributeType2
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : !distributeType2 to !strideDistributeType
    %9 = VPUIP.Copy
        inputs(%1 : !distributeType)
        outputs(%8 : !strideDistributeType)  ->  !strideDistributeType

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : !distributeType2 to !strideDistributeType
    %11 = VPUIP.Copy
        inputs(%6 : !distributeType)
        outputs(%10 : !strideDistributeType)  ->  !strideDistributeType
    %12 = VPUIP.ConcatView
        inputs(%9, %11 : !strideDistributeType, !strideDistributeType)
        outputs(%7 : !distributeType2) -> !distributeType2

    return %4, %12 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>, !distributeType2

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]

    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[BUFF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[TILINGCOPY_TO_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[ADD_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[TILINGCOPY_TO_CMX]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x32x48x176x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUFF_DDR:%.+]] = memref.alloc() : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:    [[TILINGCOPY_RESHAPE:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x32x48x176x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_DDR]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]

    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     input([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :
    // CHECK:       return  [[TILINGCOPY_RESHAPE]], [[CONCATVIEW]]
}
