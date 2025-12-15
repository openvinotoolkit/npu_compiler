//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --optimize-copies %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RemoveCMXToCMXCopyPropagateStrideToCopy
// CHECK-SAME:    [[INPUT1:%.*]]: memref<1x256x18x18xf16, #NHWC, @CMX_NN>
// CHECK-SAME:    [[INPUT2:%.*]]: memref<128x256x3x3xf16, #NHWC, @CMX_NN>

func.func @RemoveCMXToCMXCopyPropagateStrideToCopy(%arg0 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>, %arg1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>) -> (memref<1x256x18x18xf16, #NHWC, @CMX_NN>, memref<1x128x18x18xf16, #NHWC, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x128x18x18xf16, #NHWC, @CMX_NN>
    %1 = memref.alloc() : memref<1x128x18x18xf16, #NHWC, @CMX_NN>
    %2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%arg0 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) weights(%arg1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        parent_input(%arg0 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) parent_output(%0 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) outputs(%0 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) -> memref<1x128x18x18xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [17, 17, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [17, 17, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
        }

    %3 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%arg0 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) weights(%arg1 : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
        parent_input(%arg0 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) parent_output(%1 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) outputs(%1 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) -> memref<1x128x18x18xf16, #NHWC, @CMX_NN> variants : {
            DPUTask {cluster_id = 0 : i64, outEnd = [17, 17, 63], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 0]}
            DPUTask {cluster_id = 1 : i64, outEnd = [17, 17, 127], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, outStart = [0, 0, 64]}
        } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
        }
    %4 = memref.alloc() : memref<1x256x18x18xf16, #NHWC, @CMX_NN>

    %5 = VPUIP.SubView %4 [0, 0, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, #NHWC, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>
    %6 = VPUIP.Copy inputs(%2 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) outputs(%5 : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>

    %7 = VPUIP.SubView %4 [0, 128, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, #NHWC, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>
    %8 = VPUIP.Copy inputs(%3 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) outputs(%7 : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>

    %9 = VPUIP.ConcatView inputs(%6, %8 : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
            outputs(%4 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) -> memref<1x256x18x18xf16, #NHWC, @CMX_NN>
    %10 = memref.alloc() : memref<1x128x18x18xf16, #NHWC, @CMX_NN>
    %11 = VPUIP.Copy inputs(%2 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) outputs(%10 : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) -> memref<1x128x18x18xf16, #NHWC, @CMX_NN>
    return %9, %11 : memref<1x256x18x18xf16, #NHWC, @CMX_NN>, memref<1x128x18x18xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_0:%.*]] = memref.alloc() : memref<1x256x18x18xf16, #NHWC, @CMX_NN>
    // CHECK:       [[SBUVIEW_0:%.*]] = VPUIP.SubView %alloc [0, 0, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, #NHWC, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>

    // CHECK:       [[NCETASK_0:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[INPUT1]] : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) weights([[INPUT2]] : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INPUT1]] : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) parent_output([[SBUVIEW_0]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_0]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:           }
    // CHECK:       [[SBUVIEW_1:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 128, 0, 0] [1, 128, 18, 18] : memref<1x256x18x18xf16, #NHWC, @CMX_NN> to memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>
    // CHECK:       [[NCETASK_1:%.*]] = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, kernel_size = [3, 3], kernel_strides = [1, 1], minimumHardwareExecutionCost = 93417 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:      input([[INPUT1]] : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) weights([[INPUT2]] : memref<128x256x3x3xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input([[INPUT1]] : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) parent_output([[SBUVIEW_1]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[SBUVIEW_1]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) -> memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN> variants : {
    // CHECK:           DPUTask {cluster_id = 0 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           DPUTask {cluster_id = 1 : i64, mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [17, 17, 127], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    // CHECK:           } PPE : {
    // CHECK:               PPETask {ppe = #VPU.PPEStub<>}
    // CHECK:           }

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView inputs([[NCETASK_0]], [[NCETASK_1]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>, memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>)
    // CHECK-SAME:      outputs([[BUFF_0]] : memref<1x256x18x18xf16, #NHWC, @CMX_NN>) -> memref<1x256x18x18xf16, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_1:%.*]] = memref.alloc() : memref<1x128x18x18xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY2:%.*]] = VPUIP.Copy inputs([[NCETASK_0]] : memref<1x128x18x18xf16, {order = #NHWC, strides = [82944, 1, 4608, 256]}, @CMX_NN>) outputs([[BUFF_1]] : memref<1x128x18x18xf16, #NHWC, @CMX_NN>) -> memref<1x128x18x18xf16, #NHWC, @CMX_NN>

    // CHECK:       return [[CONCAT]], [[COPY2]] : memref<1x256x18x18xf16, #NHWC, @CMX_NN>, memref<1x128x18x18xf16, #NHWC, @CMX_NN>
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
// CHECK-SAME:    [[INPUT1:%.*]]: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>, [[INPUT2:%.*]]: memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>
// CHECK-SAME:    [[INPUT3:%.*]]: !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>) -> (memref<1x256x26x26x!qElemType, #NHWC, @DDR>, !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

func.func @RemoveCMXToCMXCopyPropagateStrideToQuantizeCast(%arg0 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>, %arg1 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>, %arg2 : !ConcatInputType) ->  (memref<1x256x26x26x!qElemType, #NHWC, @DDR>, !ConcatOutputType) {
    %0 = VPURT.AllocDistributed -> !ConvOutputType
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
        input(%arg0 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
        weights(%arg1 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
        parent_input(%arg0 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
        parent_output(%0 : !ConvOutputType)
        outputs(%0 : !ConvOutputType)
    ->  !ConvOutputType variants : {
        DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
    } PPE : {
        PPETask {ppe = #VPU.PPEStub<>}
    }
    %2 = VPUIP.QuantizeCast inputs(%1 : !ConvOutputType) -> memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>
    %3 = memref.alloc() : memref<1x256x26x26x!qElemType, #NHWC, @DDR>
    %4 = VPUIP.Copy
        inputs(%2 : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
        outputs(%3 : memref<1x256x26x26x!qElemType, #NHWC, @DDR>)  ->  memref<1x256x26x26x!qElemType, #NHWC, @DDR>

    %5 = VPURT.AllocDistributed -> !ConcatOutputType
    %6 = VPUIP.SubView %5 [0, 0, 0, 0] [1, 256, 26, 26] : !ConcatOutputType to !ConcatInputType
    %7 = VPUIP.Copy
        inputs(%1 : !ConvOutputType)
        outputs(%6 : !ConcatInputType)  ->  !ConcatInputType
    %8 = VPUIP.ConcatView
        inputs(%7, %arg2 : !ConcatInputType, !ConcatInputType)
        outputs(%5 : !ConcatOutputType) -> !ConcatOutputType

    return %4, %8: memref<1x256x26x26x!qElemType, #NHWC, @DDR>, !ConcatOutputType

    // CHECK:       [[CONV_OUT_ALLOC:%.*]] =  VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[CONV_OUT_SUBVIEW:%.*]] = VPUIP.SubView %0 [0, 0, 0, 0] [1, 256, 26, 26] : !VPUIP.DistributedBuffer<1x512x26x26x!qElemType, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> to !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:       [[CONV:%.*]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input([[INPUT1]] : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     weights([[INPUT2]] : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_input([[INPUT1]] : memref<1x256x26x26x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:     parent_output([[CONV_OUT_SUBVIEW]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[CONV_OUT_SUBVIEW]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK:       [[QC:%.*]] = VPUIP.QuantizeCast inputs(
    // CHECK-SAME:      [[CONV]] : !VPUIP.DistributedBuffer<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:          -> memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>
    // CHECK:       [[COPY_ALLOC:%.*]] = memref.alloc() : memref<1x256x26x26x!qElemType, #NHWC, @DDR>
    // CHECK:       [[COPY:%.*]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[QC]] : memref<1x256x26x26x!qElemType, {order = #NHWC, strides = [346112, 1, 13312, 512]}, @CMX_NN>)
    // CHECK-SAME:     outputs([[COPY_ALLOC]] : memref<1x256x26x26x!qElemType, #NHWC, @DDR>)  -> memref<1x256x26x26x!qElemType, #NHWC, @DDR>
}



// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
!qElemType1 = !quant.uniform<u8:f16, 6.7832517137714463:123>
!distributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType1 = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType2 = !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!strideDistributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
// CHECK-LABEL: @RemoveCMXToCMXClustringCopyAndInsertNewCopy
// CHECK-SAME:  [[INPUT1:%.*]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
// CHECK-SAME:  [[INPUT2:%.*]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
func.func @RemoveCMXToCMXClustringCopyAndInsertNewCopy(%arg0 : !distributeType, %arg1 : !distributeType)
                                    -> (!distributeType1, !distributeType2) {
    %0 = VPURT.AllocDistributed -> !distributeType
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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

    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)


    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[TILINGCOPY_TO_DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%2 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)  ->  memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:    [[TILINGCOPY_TO_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TILINGCOPY_TO_DDR]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFF_2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[TILINGCOPY_TO_CMX]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:              -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[BUFF_3:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[ADD_1:%.+]] =  VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[QUANTCAST]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[BUFF_3]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_3]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType1, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]

    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
!qElemType1 = !quant.uniform<u8:f16, 6.7832517137714463:123>
// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopy
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>

func.func @RemoveCMXToCMXCopyAndInsertNewCopy(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                              %arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> (memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>, memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %2 = VPUIP.QuantizeCast inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    %3 = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%2 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
            weights(%2 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
            parent_input(%2 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
            parent_output(%3 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
            outputs(%3 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %5 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%5 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%5 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %7 = memref.alloc() : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %9 = VPUIP.Copy inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%8 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%6 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%10 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>, memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                outputs(%7 : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>

    return %4, %12 : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>, memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[SUBVIEW_0:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) weights([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) parent_input([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[QUANTCAST:%.+]] = VPUIP.QuantizeCast inputs([[COPY_0]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>
    // CHECK:       [[ADD_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[QUANTCAST]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>) weights([[QUANTCAST]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>) parent_input([[QUANTCAST]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[BUFF_2]] : memref<1x64x48x88x!qElemType1, #NHWC, @CMX_NN>)

    // CHECK:       [[SUBVIEW_1:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) weights([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) parent_input([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[ADD_1]], [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeNCEUser
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
func.func @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeNCEUser(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                              %arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> (memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>, memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>
    %3 = memref.alloc() : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>
    %4 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%2 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
            weights(%2 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%2 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%3 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
            outputs(%3 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [175, 47, 31], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %5 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%5 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%5 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %7 = memref.alloc() : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %9 = VPUIP.Copy inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%8 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%6 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%10 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>, memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                outputs(%7 : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>

    return %4, %12 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>, memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[SUBVIEW_0:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) weights([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) parent_input([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[ADD_1:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[RESHAPE]] : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>) weights([[RESHAPE]] : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>) parent_input([[RESHAPE]] : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[BUFF_2]] : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)

    // CHECK:       [[SUBVIEW_1:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) weights([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) parent_input([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[ADD_1]], [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeCopyUser
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
func.func @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeCopyUser(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>,
                                              %arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                                    -> (memref<1x32x48x176x!qElemType, #NHWC, @DDR>, memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>) {
    %0 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%0 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %2 = VPUIP.GenericReshape inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>
    %3 = memref.alloc() : memref<1x32x48x176x!qElemType, #NHWC, @DDR>
    %4 = VPUIP.Copy inputs(%2 : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%3 : memref<1x32x48x176x!qElemType, #NHWC, @DDR>)
                               -> memref<1x32x48x176x!qElemType, #NHWC, @DDR>

    %5 = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
            input(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            weights(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_input(%arg1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            parent_output(%5 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
            outputs(%5 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN> variants :  {
                DPUTask {cluster_id = 0 : i64, outEnd = [87, 47, 63], mpe_mode = #VPU.mpe_mode<MATRIX>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, outStart = [0, 0, 0]}
        } PPE :  {
            PPETask {ppe = #VPU.PPEStub<>}
        }

    %7 = memref.alloc() : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>
    %8 = VPUIP.SubView %7 [0, 0, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %9 = VPUIP.Copy inputs(%1 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%8 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %10 = VPUIP.SubView %7 [0, 64, 0, 0] [1, 64, 48, 88] : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN> to memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>
    %11 = VPUIP.Copy inputs(%6 : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
                           outputs(%10 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                               -> memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>

    %12 = VPUIP.ConcatView inputs(%9, %11 : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>, memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
                outputs(%7 : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>

    return %4, %12 : memref<1x32x48x176x!qElemType, #NHWC, @DDR>, memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_0:%.+]] = memref.alloc() : memref<1x128x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[SUBVIEW_0:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) weights([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) parent_input([[INPUT1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_0:%.+]] = VPUIP.Copy inputs([[ADD_0]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)
    // CHECK-SAME:          outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[COPY_0]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) -> memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_2:%.+]] = memref.alloc() : memref<1x32x48x176x!qElemType, #NHWC, @DDR>
    // CHECK:       [[COPY_1:%.+]] = VPUIP.Copy inputs([[RESHAPE]] : memref<1x32x48x176x!qElemType, #NHWC, @CMX_NN>) outputs([[BUFF_2]] : memref<1x32x48x176x!qElemType, #NHWC, @DDR>) -> memref<1x32x48x176x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]]  = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]
    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:          input([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) weights([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>) parent_input([[INPUT2]] : memref<1x64x48x88x!qElemType, #NHWC, @CMX_NN>)
    // CHECK-SAME:          parent_output([[SUBVIEW_1]] : memref<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[COPY_1]], [[CONCATVIEW]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 5.7832517137714463:123>
!distributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType1 = !VPUIP.DistributedBuffer<1x32x48x176x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!distributeType2 = !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
!strideDistributeType = !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
// CHECK-LABEL: @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUser
// CHECK-SAME:  [[INPUT1:%.*]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>,
// CHECK-SAME:  [[INPUT2:%.*]]: !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
func.func @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUser(%arg0 : !distributeType, %arg1 : !distributeType)
                                    -> (memref<1x32x48x176x!qElemType, #NHWC, @DDR>, !distributeType2) {
    %0 = VPURT.AllocDistributed -> !distributeType
    %1 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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
    %3 = memref.alloc() : memref<1x32x48x176x!qElemType, #NHWC, @DDR>
    %4 = VPUIP.Copy
        inputs(%2 : !distributeType1)
        outputs(%3 : memref<1x32x48x176x!qElemType, #NHWC, @DDR>)  ->  memref<1x32x48x176x!qElemType, #NHWC, @DDR>

    %5 = VPURT.AllocDistributed -> !distributeType
    %6 = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
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

    return %4, %12 : memref<1x32x48x176x!qElemType, #NHWC, @DDR>, !distributeType2

    // CHECK:       [[BUFF_0:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[SUBVIEW_0:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 64, 48, 88]

    // CHECK:       [[ADD_0:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)


    // CHECK:       [[BUFF_1:%.+]] = memref.alloc() : memref<1x64x48x88x!qElemType, #NHWC, @DDR>
    // CHECK:       [[TILINGCOPY_TO_DDR:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs(%2 : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_1]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)  ->  memref<1x64x48x88x!qElemType, #NHWC, @DDR>

    // CHECK:       [[BUFF_2:%.+]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[TILINGCOPY_TO_CMX:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[TILINGCOPY_TO_DDR]] : memref<1x64x48x88x!qElemType, #NHWC, @DDR>)
    // CHECK-SAME:     outputs([[BUFF_2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)  ->  !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>

    // CHECK:       [[RESHAPE:%.+]] = VPUIP.GenericReshape inputs([[TILINGCOPY_TO_CMX]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>) -> !VPUIP.DistributedBuffer<1x32x48x176x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>
    // CHECK:       [[BUFF_DDR:%.+]] = memref.alloc() : memref<1x32x48x176x!qElemType, #NHWC, @DDR>
    // CHECK:    [[TILINGCOPY_RESHAPE:%.+]] = VPUIP.Copy
    // CHECK-SAME:     inputs([[RESHAPE]] : !VPUIP.DistributedBuffer<1x32x48x176x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[BUFF_DDR]] : memref<1x32x48x176x!qElemType, #NHWC, @DDR>)  ->  memref<1x32x48x176x!qElemType, #NHWC, @DDR>

    // CHECK:       [[SUBVIEW_1:%.+]] = VPUIP.SubView [[BUFF_0]] [0, 64, 0, 0] [1, 64, 48, 88]



    // CHECK:       [[ADD_2:%.+]] = VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 1081 : i64, task_type = #VPUIP.nce_task_type<ELTWISE>}
    // CHECK-SAME:     input([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     weights([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_input([[INPUT2]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     parent_output([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1]] : !VPUIP.DistributedBuffer<1x64x48x88x!qElemType, {order = #NHWC, strides = [540672, 1, 11264, 128]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 4, 1], num_clusters = 4 : i64}>)

    // CHECK:       [[CONCATVIEW:%.+]] = VPUIP.ConcatView inputs([[ADD_0]], [[ADD_2]] :

    // CHECK:       return  [[TILINGCOPY_RESHAPE]], [[CONCATVIEW]]
}

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, #NHWC, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, #NHWC, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, #NHWC, @CMX_NN>

!OutDataBufferType = !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
!OutSMBufferType = !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

// CHECK-LABEL: @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUserSparsity
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x256x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.*]]: memref<128x256x3x3xf16, #NHWC, @CMX_NN>
func.func @RemoveCMXToCMXTilingCopyAndInsertNewCopyWithReshapeCopyUserSparsity(
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType)
    -> (!OutDataBufferType, !OutSMBufferType, memref<1x128x28x7xf16, #NHWC, @DDR>, memref<1x128x28x7xi1, #NHWC, @DDR>)
{
    // alloc for Conv data out
    %0 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // alloc for Conv sparsity map out
    %1 = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
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
    %12:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
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

    %17 = memref.alloc() : memref<1x128x28x7xf16, #NHWC, @DDR>
    %18 = memref.alloc() : memref<1x128x28x7xi1, #NHWC, @DDR>

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
        outputs(%17 : memref<1x128x28x7xf16, #NHWC, @DDR>)  ->  memref<1x128x28x7xf16, #NHWC, @DDR>
    %sm21 = VPUIP.Copy
        inputs(%sm20 : !VPUIP.DistributedBuffer<1x128x28x7xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
        outputs(%18 : memref<1x128x28x7xi1, #NHWC, @DDR>)  ->  memref<1x128x28x7xi1, #NHWC, @DDR>

    return %data16, %sm16, %data21, %sm21 : !OutDataBufferType, !OutSMBufferType, memref<1x128x28x7xf16, #NHWC, @DDR>, memref<1x128x28x7xi1, #NHWC, @DDR>

    // CHECK:      [[BUFF_1:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[BUFF_0:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x256x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SUBVIEW_0_DATA:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SUBVIEW_0_SM:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[NCE_0:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:     parent_output([[SUBVIEW_0_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     parent_output_sparsity_map([[SUBVIEW_0_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_0_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     output_sparsity_map([[SUBVIEW_0_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)

    // CHECK:      [[SUBVIEW_1_DATA:%.*]] = VPUIP.SubView [[BUFF_0]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[SUBVIEW_1_SM:%.*]] = VPUIP.SubView [[BUFF_1]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:  to !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>

    // CHECK:      [[NCE_1:%.*]]:2 = VPUIP.NCEClusterTask {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
    // CHECK-SAME:     parent_output([[SUBVIEW_1_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     parent_output_sparsity_map([[SUBVIEW_1_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     outputs([[SUBVIEW_1_DATA]] : !VPUIP.DistributedBuffer<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)
    // CHECK-SAME:     output_sparsity_map([[SUBVIEW_1_SM]] : !VPUIP.DistributedBuffer<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>)


    // CHECK:      [[BUFF_3:%.*]] = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @DDR>
    // CHECK:      [[COPY_TO_DDR_SM:%.*]] = VPUIP.Copy inputs([[NCE_1]]#1
    // CHECK-SAME:  outputs([[BUFF_3]]
    // CHECK:       [[BUFF_5:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:       [[COPY_TO_CMX_SM:%.*]] = VPUIP.Copy inputs([[COPY_TO_DDR_SM]]
    // CHECK-SAME:  outputs([[BUFF_5]]

    // CHECK:      [[BUFF_2:%.*]] = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @DDR>
    // CHECK:      [[COPY_TO_DDR_DATA:%.*]] = VPUIP.Copy inputs([[NCE_1]]#0
    // CHECK-SAME:  outputs([[BUFF_2]]
    // CHECK:       [[BUFF_4:%.*]] = VPURT.AllocDistributed -> !VPUIP.DistributedBuffer<1x128x14x14xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[COPY_TO_CMX_DATA:%.*]] = VPUIP.Copy inputs([[COPY_TO_DDR_DATA]]
    // CHECK-SAME:  outputs([[BUFF_4]]

    // CHECK:      [[CONCAT_DATA:%.*]] =  VPUIP.ConcatView inputs([[NCE_0]]#0, [[NCE_1]]#0
    // CHECK-SAME:  outputs([[BUFF_0]]
    // CHECK:      [[CONCAT_SM:%.*]] =  VPUIP.ConcatView inputs([[NCE_0]]#1, [[NCE_1]]#1
    // CHECK-SAME:  outputs([[BUFF_1]]

    // CHECK:      [[BUFF_6:%.*]]  = memref.alloc() : memref<1x128x28x7xf16, #NHWC, @DDR>
    // CHECK:      [[BUFF_7:%.*]] = memref.alloc() : memref<1x128x28x7xi1, #NHWC, @DDR>
    // CHECK:      [[RESHAPE_DATA:%.*]] = VPUIP.GenericReshape inputs([[COPY_TO_CMX_DATA]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x128x28x7xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[RESHAPE_SM:%.*]] = VPUIP.GenericReshape inputs([[COPY_TO_CMX_SM]]
    // CHECK-SAME:  -> !VPUIP.DistributedBuffer<1x128x28x7xi1, #NHWC, @CMX_NN, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    // CHECK:      [[COPY_OUT_DATA:%.*]] = VPUIP.Copy inputs([[RESHAPE_DATA]]
    // CHECK-SAME:  outputs([[BUFF_6]]
    // CHECK:      [[COPY_OUT_SM:%.*]] = VPUIP.Copy inputs([[RESHAPE_SM]]
    // CHECK-SAME:  outputs([[BUFF_7]]

    // CHECK:       return [[CONCAT_DATA]], [[CONCAT_SM]], [[COPY_OUT_DATA]], [[COPY_OUT_SM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InDataType = memref<1x256x28x28xf16, #NHWC, @CMX_NN>
!InSMType = memref<1x256x28x28xi1, #NHWC, @CMX_NN>
!ConvWeightsType = memref<128x256x3x3xf16, #NHWC, @CMX_NN>

!OutDataBufferType = memref<1x256x14x14xf16, #NHWC, @CMX_NN>
!OutSMBufferType = memref<1x256x14x14xi1, #NHWC, @CMX_NN>

// CHECK-LABEL: @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeSparsity
// CHECK-SAME:  [[INPUT1:%.*]]: memref<1x256x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT2:%.*]]: memref<1x256x28x28xi1, #NHWC, @CMX_NN>,
// CHECK-SAME:  [[INPUT3:%.*]]: memref<128x256x3x3xf16, #NHWC, @CMX_NN>
func.func @RemoveCMXToCMXCopyAndInsertNewCopyWithReshapeSparsity(
    %inData : !InDataType,
    %inSparsityMap : !InSMType,
    %inWeights : !ConvWeightsType)
    -> (!OutDataBufferType, !OutSMBufferType, memref<1x128x7x28xf16, #NHWC, @CMX_NN>, memref<1x128x7x28xi1, #NHWC, @CMX_NN>)
{
    // alloc for Conv data out
    %0 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %1 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %2:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
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
    %data8 = VPUIP.Copy inputs(%2#0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
            outputs(%data7 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %sm8 = VPUIP.Copy inputs(%2#1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
            outputs(%sm7 : memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %data9 = VPUIP.GenericReshape inputs(%2#0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        -> memref<1x128x7x28xf16, #NHWC, @CMX_NN>

    %sm9 = VPUIP.GenericReshape inputs(%2#1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x7x28xi1, #NHWC, @CMX_NN>

    // alloc for Conv data out
    %10 = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // alloc for Conv sparsity map out
    %11 = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>

    // Input 1: Convolution
    %12:2 = VPUIP.NCEClusterTask
            {kernel_padding = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, kernel_size = [3, 3], kernel_strides = [2, 2], minimumHardwareExecutionCost = 34660 : i64, task_type = #VPUIP.nce_task_type<CONV>}
        input(%inData : !InDataType)
        input_sparsity_map(%inSparsityMap : !InSMType)
        weights(%inWeights : !ConvWeightsType)
        parent_input(%inData : !InDataType)
        parent_input_sparsity_map(%inSparsityMap : !InSMType)
        parent_output(%10 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        parent_output_sparsity_map(%11 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        outputs(%10 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
        output_sparsity_map(%11 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
        -> memref<1x128x14x14xf16, #NHWC, @CMX_NN>, memref<1x128x14x14xi1, #NHWC, @CMX_NN>
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
    %data15 = VPUIP.Copy inputs(%12#0 : memref<1x128x14x14xf16, #NHWC, @CMX_NN>)
            outputs(%data14 : memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>)
            -> memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>

    %sm15 = VPUIP.Copy inputs(%12#1 : memref<1x128x14x14xi1, #NHWC, @CMX_NN>)
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
        memref<1x128x7x28xf16, #NHWC, @CMX_NN>, memref<1x128x7x28xi1, #NHWC, @CMX_NN>

    // CHECK:       [[BUFF_SM:%.+]] = memref.alloc() : memref<1x256x14x14xi1, #NHWC, @CMX_NN>
    // CHECK:       [[BUFF_DATA:%.+]] = memref.alloc() : memref<1x256x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       [[SUBVIEW_DATA:%.+]] = VPUIP.SubView [[BUFF_DATA]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:      to memref<1x128x14x14xf16, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    // CHECK:       [[SUBVIEW_SM:%.+]] = VPUIP.SubView [[BUFF_SM]] [0, 0, 0, 0] [1, 128, 14, 14]
    // CHECK-SAME:      to memref<1x128x14x14xi1, {order = #NHWC, strides = [50176, 1, 3584, 256]}, @CMX_NN>
    // CHECK:       [[CLUST_TASK:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      output([[SUBVIEW_DATA]]
    // CHECK-SAME:      output_sparsity_map([[SUBVIEW_SM]]

    // CHECK:       [[BUFF_SM1:%.+]] = memref.alloc() : memref<1x128x14x14xi1, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_OUT_SM:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CLUST_TASK]]#1
    // CHECK-SAME:      outputs([[BUFF_SM1]]
    // CHECK:       [[BUFF_DATA1:%.+]] = memref.alloc() : memref<1x128x14x14xf16, #NHWC, @CMX_NN>
    // CHECK:       [[COPY_OUT_DATA:%.+]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[CLUST_TASK]]#0
    // CHECK-SAME:      outputs([[BUFF_DATA1]]

    // CHECK:       [[RESHAPE_DATA:%.+]] = VPUIP.GenericReshape inputs([[COPY_OUT_DATA]]
    // CHECK-SAME:      -> memref<1x128x7x28xf16, #NHWC, @CMX_NN>
    // CHECK:       [[RESHAPE_SM:%.+]] = VPUIP.GenericReshape inputs([[COPY_OUT_SM]]
    // CHECK-SAME:      -> memref<1x128x7x28xi1, #NHWC, @CMX_NN>

    // CHECK:       [[SUBVIEW1_DATA:%.+]] = VPUIP.SubView [[BUFF_DATA]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK:       [[SUBVIEW1_SM:%.+]] = VPUIP.SubView [[BUFF_SM]] [0, 128, 0, 0] [1, 128, 14, 14]
    // CHECK:       [[CLUST_TASK1:%.+]]:2 = VPUIP.NCEClusterTask
    // CHECK-SAME:      output([[SUBVIEW1_DATA]]
    // CHECK-SAME:      output_sparsity_map([[SUBVIEW1_SM]]

    // CHECK:       [[CONCAT_DATA:%.+]] =  VPUIP.ConcatView inputs([[CLUST_TASK]]#0, [[CLUST_TASK1]]#0
    // CHECK:       [[CONCAT_SM:%.+]] =  VPUIP.ConcatView inputs([[CLUST_TASK]]#1, [[CLUST_TASK1]]#1

    // CHECK:       return [[CONCAT_DATA]], [[CONCAT_SM]], [[RESHAPE_DATA]], [[RESHAPE_SM]]
}
