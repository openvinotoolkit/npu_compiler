//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --insert-delay-dpu-variant %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>

// CHECK-LABEL: @DPUTaskWithoutSprLUT
func.func private @DPUTaskWithoutSprLUT(%input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%input : !DataType)
                        weights(%weights : !WeightsType)
                        weight_table(%weight_table : !WeightTableType)
                        parent_input(%input : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK-NOT: DPUTask

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvWithSprLUTKernel1x1
func.func private @ConvWithSprLUTKernel1x1(%input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%input : !DataType)
                        weights(%weights : !WeightsType)
                        weight_table(%weight_table : !WeightTableType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [0, 0, 15]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 15]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvWithSprLUTKernel3x3
func.func private @ConvWithSprLUTKernel3x3(%input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [3, 3],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%input : !DataType)
                        weights(%weights : !WeightsType)
                        weight_table(%weight_table : !WeightTableType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [2, 2, 15]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 15]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvWithSprLUTKernel3x3Padding1x1x1x1
func.func private @ConvWithSprLUTKernel3x3Padding1x1x1x1(%input: !DataType, %weights: !WeightsType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                            kernel_size = [3, 3],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%input : !DataType)
                        weights(%weights : !WeightsType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [1, 1, 15]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 15]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x4x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<3x4x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvWithSprLUTKernel1x1Autopad
func.func private @ConvWithSprLUTKernel1x1Autopad(%input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<CONV>
                        }
                        input(%input : !DataType)
                        weights(%weights : !WeightsType)
                        weight_table(%weight_table : !WeightTableType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 3], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 2], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [0, 0, 3]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 2]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 3]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 2]
// CHECK-DAG: outStart = [0, 0, 0]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x64x3x4xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<1x64x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<16x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
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

// CHECK-LABEL: @ConvWithODUAutopadAndHalo
func.func  @ConvWithODUAutopadAndHalo(
        %input: !DataType, %weights: !WeightsType, %weight_table: !WeightTableType,
        %output_iti0: !OutputITICluster0, %output_iti1: !OutputITICluster1, %output_iti2: !OutputITICluster2) -> !OutputITICluster0 {
    %nce = VPUIP.NCEClusterTask {
            is_superdense,
            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            task_type = #VPUIP.nce_task_type<CONV>
        } input(%input : !DataType)
          weights(%weights : !WeightsType)
          weight_table(%weight_table : !WeightTableType)
          parent_input(%input : memref<1x64x3x4xf16, #NHWC, [@CMX_NN, 0]>)
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

    // CHECK:       DPUTask
    // CHECK-SAME:  cluster_id = 0 : i64,
    // CHECK-NOT:   haloRegions
    // CHECK-SAME:  inEnd = [0, 0, 63],
    // CHECK-SAME:  inStart = [0, 0, 0],
    // CHECK-SAME:  mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
    // CHECK-SAME:  outEnd = [0, 0, 0],
    // CHECK-SAME:  outStart = [0, 0, 0],
    // CHECK-SAME:  pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @EltwiseAddWithSprLUT
func.func private @EltwiseAddWithSprLUT(%input1: !DataType, %input2: !DataType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            eltwise_type = #VPU.eltwise_type<ADD>,
                            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                            task_type = #VPUIP.nce_task_type<ELTWISE>
                        }
                        input(%input1 : !DataType)
                        weights(%input2 : !DataType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input1 : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [0, 0, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 31]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @MaxPoolWithSprLUT
func.func private @MaxPoolWithSprLUT(%input1: !DataType, %weight_table: !WeightTableType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [2, 2],
                            kernel_strides = [2, 2],
                            task_type = #VPUIP.nce_task_type<MAXPOOL>
                        }
                        input(%input1 : !DataType)
                        weight_table(%weight_table : !WeightTableType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input1 : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [1, 1, 15]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 15]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DataType = memref<1x32x16x16xf16, #NHWC, [@CMX_NN, 0]>
!WeightsType = memref<32x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
!WeightTableType = memref<32x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
!SrplutType = memref<512xui16, [@CMX_NN, 0]>

// CHECK-LABEL: @ConvWithSprLUTAndProfiling
func.func private @ConvWithSprLUTAndProfiling(%input: !DataType, %weights: !WeightsType, %sprlut: !SrplutType, %output: !DataType) -> !DataType {
    %dpu_output = VPUIP.NCEClusterTask {
                            kernel_padding = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                            kernel_size = [1, 1],
                            kernel_strides = [1, 1],
                            task_type = #VPUIP.nce_task_type<CONV>,
                            profilingMetadata = #VPUIP.DpuProfilingMetadataAttr<bufferId = 1 : i64, taskId = 1 : i64, maxVariants = 1 : i64>
                        }
                        input(%input : !DataType)
                        weights(%weights : !WeightsType)
                        spr_lookup_table(%sprlut: !SrplutType)
                        parent_input(%input : !DataType)
                        parent_output(%output : !DataType)
                        outputs(%output : !DataType) -> !DataType
                        variants : {
        DPUTask {inEnd = [15, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 15, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 42 : i64}
    } PPE : {
    }
    return %dpu_output : !DataType
}

// CHECK:     DPUTask
// CHECK-DAG: inEnd = [0, 0, 15]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [0, 0, 15]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
// CHECK-DAG: workload_id = 42
// CHECK:     DPUTask
// CHECK-DAG: inEnd = [15, 15, 31]
// CHECK-DAG: inStart = [0, 0, 0]
// CHECK-DAG: outEnd = [15, 15, 31]
// CHECK-DAG: outStart = [0, 0, 0]
// CHECK-DAG: workload_id = 42
