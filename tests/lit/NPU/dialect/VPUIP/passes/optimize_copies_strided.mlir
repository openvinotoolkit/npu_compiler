//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --optimize-copies %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveCMXToCMXCopyPropagateStrideToCopy
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME:    [[INPUT2:%.+]]: memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>

func.func @RemoveCMXToCMXCopyPropagateStrideToCopy(%arg0 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>, %arg1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>) -> (memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>
    %1 = memref.alloc() : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%arg0 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) weights(%arg1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%arg0 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) parent_output(%0 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) outputs(%0 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [17, 17, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [17, 17, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
        }

    %3 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], task_type = #VPUIP.nce_task_type<CONV>}>
        input(%arg0 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) weights(%arg1 : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
        parent_input(%arg0 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) parent_output(%1 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) outputs(%1 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [17, 17, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [17, 17, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
        }
    %4 = memref.alloc() : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>
    %6 = VPUIP.Copy inputs(%2 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) outputs(%5 : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>

    %7 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>
    %8 = VPUIP.Copy inputs(%3 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) outputs(%7 : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>

    %9 = VPUIP.ConcatView inputs(%6, %8 : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
            outputs(%4 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>
    %10 = memref.alloc() : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%2 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) outputs(%10 : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>
    return %9, %11 : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[SBUVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>

    // CHECK:       [[NCETASK_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1]
    // CHECK-SAME:      input([[INPUT1]] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) weights([[INPUT2]] : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INPUT1]] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) parent_output([[SBUVIEW_0]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_0]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:           }
    // CHECK:       [[SBUVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 128, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>
    // CHECK:       [[NCETASK_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 93417 : i64} <{kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1]
    // CHECK-SAME:      input([[INPUT1]] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) weights([[INPUT2]] : memref<128x256x3x3xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INPUT1]] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) parent_output([[SBUVIEW_1]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_1]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:           }

    // CHECK:       [[CONCAT:%.+]] = VPUIP.ConcatView inputs([[NCETASK_0]], [[NCETASK_1]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY2:%.+]] = VPUIP.Copy inputs([[NCETASK_0]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) outputs([[BUFF_1]] : memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       return [[CONCAT]], [[COPY2]] : memref<1x256x18x18xf16, {order = #NHWC}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.0:123>
!qElemType1 = !quant.uniform<u8:f16, 2.0:123>

!ConcatInputType = !VPUIP.DistributedBuffer<
    1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64,
    alignment = [1, 16, 1, 1]
}>

!ConvOutputType = !VPUIP.DistributedBuffer<
    1x256x26x26x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>

!ConcatOutputType = !VPUIP.DistributedBuffer<
    1x512x26x26x!qElemType, #NHWC, @CMX_NN, {
    mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1],
    num_clusters = 2 : i64, alignment = [1, 16, 1, 1]
}>

!QCOutType = !VPUIP.DistributedBuffer<
    1x256x26x26x!qElemType1, #NHWC, @CMX_NN, {
    mode = "SEGMENTED", num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

// CHECK-LABEL: @RemoveCMXToCMXCopyPropagateStrideToQuantizeCast
// CHECK-SAME:    [[INPUT1:%.+]]: memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>, [[INPUT2:%.+]]: memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME:    [[INPUT3:%.+]]: !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> (memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>, !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

func.func @RemoveCMXToCMXCopyPropagateStrideToQuantizeCast(%arg0 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>, %arg1 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>, %arg2 : !ConcatInputType) ->  (memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>, !ConcatOutputType) {
    %0 = VPURT.AllocDistributed -> !ConvOutputType
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
        input(%arg0 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
        weights(%arg1 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_input(%arg0 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
        parent_output(%0 : !ConvOutputType)
        outputs(%0 : !ConvOutputType)
    ->  !ConvOutputType variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %2 = VPUIP.QuantizeCast inputs(%1 : !ConvOutputType) -> memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
        outputs(%3 : memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>)  ->  memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>

    %5 = VPURT.AllocDistributed -> !ConcatOutputType
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 256, 26, 26] : !ConcatOutputType to !ConcatInputType
    %7 = VPUIP.Copy
        inputs(%1 : !ConvOutputType)
        outputs(%6 : !ConcatInputType)  ->  !ConcatInputType
    %8 = VPUIP.ConcatView
        inputs(%7, %arg2 : !ConcatInputType, !ConcatInputType)
        outputs(%5 : !ConcatOutputType) -> !ConcatOutputType

    return %4, %8: memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>, !ConcatOutputType

    // CHECK:       [[CONV_OUT_ALLOC:%.+]] =  VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[CONV_OUT_SUBVIEW:%.+]] = VPUIP.SubView [[CONV_OUT_ALLOC]] [0, 0, 0, 0] [1, 256, 26, 26] : !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[CONV:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:     input([[INPUT1]] : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     weights([[INPUT2]] : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_input([[INPUT1]] : memref<1x256x26x26x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:     parent_output([[CONV_OUT_SUBVIEW]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[CONV_OUT_SUBVIEW]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK:       [[QC:%.+]] = VPUIP.QuantizeCast inputs(
    // CHECK-SAME:      [[CONV]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:          -> memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>
    // CHECK:       [[COPY_ALLOC:%.+]] = memref.alloc() : memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QC]] : memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>)
    // CHECK-SAME:     outputs([[COPY_ALLOC]] : memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>)  -> memref<1x256x26x26x!qElemType, {order = #NHWC}, @DDR>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
!qElemType1 = !quant.uniform<u8:f16, 6.7832517137714463:123>
// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopy
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

func.func @RemoveCMXToCMXCopyAndInsertNewCopy(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
                                              %arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                                    -> (memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>, memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%2 : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)
            weights(%2 : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)
            parent_input(%2 : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)
            parent_output(%3 : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)
            outputs(%3 : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %5 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%5 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%5 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %7 = memref.alloc() : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %9 = VPUIP.Copy inputs(%1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%8 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%6 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%10 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>, memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                outputs(%7 : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

    return %4, %12 : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>, memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[SUBVIEW_0:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[COPY_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[ADD_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[QUANTCAST]] : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>) weights([[QUANTCAST]] : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>) parent_input([[QUANTCAST]] : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[BUFF_2]] : memref<1x64x48x88x!qElemType1, {order = #NHWC}, @CMX_NN>)

    // CHECK:       [[SUBVIEW_1:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[ADD_1]], [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeNCEUser
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeNCEUser(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
                                              %arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                                    -> (memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>, memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
      <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
      <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%2 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%2 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%2 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%3 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%3 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [175, 47, 31], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %5 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%5 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%5 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %7 = memref.alloc() : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %9 = VPUIP.Copy inputs(%1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%8 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%6 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%10 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>, memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                outputs(%7 : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

    return %4, %12 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>, memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[SUBVIEW_0:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[ADD_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[RESHAPE]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[RESHAPE]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[RESHAPE]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[BUFF_2]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)

    // CHECK:       [[SUBVIEW_1:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[ADD_1]], [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeCopyUser
// CHECK-SAME:  [[INPUT1:%.+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.+]]: memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeCopyUser(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>,
                                              %arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                                    -> (memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>, memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%arg0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%0 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>
    %3 = memref.alloc() : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%3 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>)
                               -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>

    %5 = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>}
     <{task_type = #VPUIP.nce_task_type<ELTWISE>}>
            input(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            weights(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_input(%arg1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            parent_output(%5 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
            outputs(%5 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %7 = memref.alloc() : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %9 = VPUIP.Copy inputs(%1 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%8 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%6 : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
                           outputs(%10 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>, memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                outputs(%7 : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

    return %4, %12 : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>, memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[SUBVIEW_0:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[INPUT1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>
    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @CMX_NN>) outputs([[BUFF_2]] : memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>) -> memref<1x32x48x176x!qElemType, {order = #NHWC}, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64}
    // CHECK-SAME:          input([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) weights([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>) parent_input([[INPUT2]] : memref<1x64x48x88x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[COPY_1]], [[CONCATVIEW]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// SubView with a non-trivial stride on dim0 (e.g. stride=[4,1,1,1] on shape [4352,1,1,1])
// reads every 4th element of the source. The resulting Copy must NOT be replaced with a ViewOp:
// ViewOp would reinterpret the strided layout as contiguous, reading wrong data.

// CHECK-LABEL: func.func @DoNotReplaceOutStridedSubViewCopyWithViewOp
// CHECK-SAME:      [[ARG0:%.+]]: memref<17408x1x1x1xf16, @DDR>
func.func @DoNotReplaceOutStridedSubViewCopyWithViewOp(%arg0: memref<17408x1x1x1xf16, @DDR>) -> memref<4352x1x1x1xf16, @DDR> {
    %0 = memref.alloc() : memref<4352x1x1x1xf16, @DDR>
    %1 = VPUIP.SubView %arg0 [0, 0, 0, 0] [4352, 1, 1, 1] [4, 1, 1, 1] : memref<17408x1x1x1xf16, @DDR> to memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>
    %2 = VPUIP.Copy inputs(%1 : memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>) outputs(%0 : memref<4352x1x1x1xf16, @DDR>) -> memref<4352x1x1x1xf16, @DDR>
    return %2 : memref<4352x1x1x1xf16, @DDR>

    // CHECK-NOT: VPUIP.ViewOp
    // CHECK:     [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [4352, 1, 1, 1] [4, 1, 1, 1]
    // CHECK-SAME:    memref<17408x1x1x1xf16, @DDR> to memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>
    // CHECK:     [[COPY:%.+]] = VPUIP.Copy inputs([[SUBVIEW]] : memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>)
    // CHECK-SAME:    outputs({{%.+}} : memref<4352x1x1x1xf16, @DDR>) -> memref<4352x1x1x1xf16, @DDR>
    // CHECK:     return [[COPY]] : memref<4352x1x1x1xf16, @DDR>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @DoNotReplaceStridedInputSubViewCopyWithViewOp
// CHECK-SAME: ([[ARG0:%.+]]: memref<17408x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>)
func.func @DoNotReplaceStridedInputSubViewCopyWithViewOp(
    %arg0: memref<17408x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>)
    -> memref<4352x1x1x1xf16, @DDR> {

    %out = memref.alloc() : memref<4352x1x1x1xf16, @DDR>

    %sub = VPUIP.SubView %arg0 [0, 0, 0, 0] [4352, 1, 1, 1]
        : memref<17408x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>
          to memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>

    %copy = VPUIP.Copy
        inputs(%sub : memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>)
        outputs(%out : memref<4352x1x1x1xf16, @DDR>)
        -> memref<4352x1x1x1xf16, @DDR>

    return %copy : memref<4352x1x1x1xf16, @DDR>

    // CHECK-NOT: VPUIP.ViewOp

    // CHECK: [[SUBVIEW:%.+]] = VPUIP.SubView [[ARG0]] [0, 0, 0, 0] [4352, 1, 1, 1]
    // CHECK-SAME: memref<17408x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>
    // CHECK-SAME: to memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>

    // CHECK: [[COPY:%.+]] = VPUIP.Copy
    // CHECK-SAME: inputs([[SUBVIEW]] : memref<4352x1x1x1xf16, {order = #NCHW, strides = [4, 1, 1, 1]}, @DDR>)
    // CHECK-SAME: outputs({{%.+}} : memref<4352x1x1x1xf16, @DDR>) -> memref<4352x1x1x1xf16, @DDR>

    // CHECK: return [[COPY]] : memref<4352x1x1x1xf16, @DDR>
}
