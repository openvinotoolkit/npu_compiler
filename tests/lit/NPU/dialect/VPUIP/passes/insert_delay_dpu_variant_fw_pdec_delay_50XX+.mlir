//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --insert-delay-dpu-variant="fw-pdec-delay-enabled=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x64x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightsType = memref<1x64x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
!WeightTableType = memref<16x1x1x4xsi32, {order = #NHWC}, [@CMX_NN, 0]>
!OutputITICluster0 = !VPUIP.ITIBuffer<
    1x1x8x4xf16, #NHWC, [@CMX_NN, 0],
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 3, 0], cluster_id = 0 : i64>,
        #VPUIP.HaloRegionAttr<shape = [1, 1, 2, 4], offset = [0, 0, 6, 0], cluster_id = 0 : i64>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
            #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
        ]>
]>
!OutputITICluster1 = !VPUIP.ITIBuffer<
    1x1x8x4xf16, #NHWC, [@CMX_NN, 1],
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
        #VPUIP.HaloRegionAttr<shape = [1, 1, 2, 4], offset = [0, 0, 6, 0], cluster_id = 1 : i64>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 3, 0], cluster_id = 1 : i64, inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 3, 0], cluster_id = 0 : i64>,
            #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 3, 0], cluster_id = 2 : i64>
        ]>
]>
!OutputITICluster2 = !VPUIP.ITIBuffer<
    1x1x8x4xf16, #NHWC, [@CMX_NN, 2],
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
        #VPUIP.HaloRegionAttr<shape = [1, 1, 3, 4], offset = [0, 0, 3, 0], cluster_id = 2 : i64>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 1, 2, 4], offset = [0, 0, 6, 0], cluster_id = 2 : i64, inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 1, 2, 4], offset = [0, 0, 6, 0], cluster_id = 0 : i64>,
            #VPUIP.HaloRegionAttr<shape = [1, 1, 2, 4], offset = [0, 0, 6, 0], cluster_id = 1 : i64>
        ]>
]>

// CHECK-LABEL: @SkipConvWithODUAutopadAndHaloFwPdecDelay
func.func  @SkipConvWithODUAutopadAndHaloFwPdecDelay(
        %input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType,
        %output_iti0: !OutputITICluster0, %output_iti1: !OutputITICluster1, %output_iti2: !OutputITICluster2) -> !OutputITICluster0 {
    %nce = VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>} <{
            is_superdense,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            task_type = #VPUIP.nce_task_type<CONV>
        }> input(%input : !DataType)
          weights(%weights : !WeightsType)
          weight_table(%weight_table : !WeightTableType)
          parent_input(%input : memref<1x64x3x4xf16, {order = #NHWC}, [@CMX_NN, 0]>)
          parent_output(%output_iti0 : !OutputITICluster0)
          output_ITI_buff(%output_iti1, %output_iti2 : !OutputITICluster1, !OutputITICluster2)
          outputs(%output_iti0 : !OutputITICluster0) -> !OutputITICluster0
          variants : {
            DPUTask {
                cluster_id = 0 : i64,
                haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 3 : i64, yStart = 0 : i64, yEnd = 2 : i64, zStart = 0 : i64, zEnd = 0 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 4 : i64>],
                inEnd = [3, 2, 63],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                outEnd = [3, 2, 0],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
            }
    } PPE : {
    }
    return %nce : !OutputITICluster0

    // CHECK:       DPUTask
    // CHECK-SAME:  cluster_id = 0 : i64,
    // CHECK-SAME:  haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 3 : i64, yStart = 0 : i64, yEnd = 2 : i64, zStart = 0 : i64, zEnd = 0 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 4 : i64>],
    // CHECK-SAME:  inEnd = [3, 2, 63],
    // CHECK-SAME:  inStart = [0, 0, 0],
    // CHECK-SAME:  mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
    // CHECK-SAME:  outEnd = [3, 2, 0],
    // CHECK-SAME:  outStart = [0, 0, 0],
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>

    // CHECK-NOT:   DPUTask
}
