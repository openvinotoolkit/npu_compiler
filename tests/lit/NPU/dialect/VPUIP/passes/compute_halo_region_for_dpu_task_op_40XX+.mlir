//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true compilation-mode=DefaultHW" --compute-halo-region-for-dpu-task-op %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x16x32xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x16x32xf16, #NHWC, [@CMX_NN, 1]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x16x17x32xf16, #NHWC, [@CMX_NN, 0], // top half of height
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 15, 0], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>]>]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x16x17x32xf16, #NHWC, [@CMX_NN, 1], // bottom half of height
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 1, 0], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>]>]>

// CHECK-LABEL: @TwoNCEClusterTasksSOH
module @TwoNCEClusterTasksSOH {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x17x32xf16>
        DataInfo "input1" : tensor<1x16x17x32xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x16x17x32xf16>
        DataInfo "output1" : tensor<1x16x17x32xf16>
    }

func.func @main(%arg0:  memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>, %arg1:  memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>) -> (memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> ->  !OutputITI1

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1 : !OutputITI1)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over H
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 15],
                    inStart = [0, 0, 0],
                    inEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 8, 0],
                    outEnd = [31, 15, 15],
                    inStart = [0, 8, 0],
                    inEnd = [31, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0: !OutputITI0)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over H
                DPUTask {
                    outStart = [0, 1, 0],
                    outEnd = [31, 8, 15],
                    inStart = [0, 0, 0],
                    inEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [0, 9, 0],
                    outEnd = [31, 16, 15],
                    inStart = [0, 8, 0],
                    inEnd = [31, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    return %arg0, %arg1: memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:                haloRegions = [],
// CHECK-SAME:               outEnd = [31, 7, 15],
// CHECK-SAME:               outStart = [0, 0, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 31 : i64, yStart = 15 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -15360 : i64, targetClusters = [1], targetWidth = 32 : i64>],
// CHECK-SAME:               inEnd = [31, 15, 15],
// CHECK-SAME:               inStart = [0, 8, 0],
// CHECK-SAME:               outEnd = [31, 15, 15],
// CHECK-SAME:               outStart = [0, 8, 0],

// CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 31 : i64, yStart = 1 : i64, yEnd = 1 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 15360 : i64, targetClusters = [0], targetWidth = 32 : i64>],
// CHECK-SAME:               outEnd = [31, 8, 15],
// CHECK-SAME:               outStart = [0, 1, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               haloRegions = [],
// CHECK-SAME:               inEnd = [31, 15, 15],
// CHECK-SAME:               inStart = [0, 8, 0],
// CHECK-SAME:               outEnd = [31, 16, 15],
// CHECK-SAME:               outStart = [0, 9, 0],

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x32x17xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x32x17xf16, #NHWC, [@CMX_NN, 1]
>

!Input2 = memref<
    1x16x32x17xf16, #NHWC, [@CMX_NN, 2]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x16x32x18xf16, #NHWC, [@CMX_NN, 0], // left-most slice of W
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 17], cluster_id = 0>],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 0, 16], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 0, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 16, 16], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 16, 0], cluster_id = 1>]>
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x16x32x19xf16, #NHWC, [@CMX_NN, 1], // middle slice of W
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 16, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 18], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 1], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 17], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 17], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 0], cluster_id = 2>]>
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x16x32x18xf16, #NHWC, [@CMX_NN, 2], // right-most slice of W
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 0], cluster_id = 2>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 1], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 32, 1], offset = [0, 0, 0, 18], cluster_id = 1>]>]>

// CHECK-LABEL: @ThreeNCEClusterTasksSOW
module @ThreeNCEClusterTasksSOW {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x32x17xf16>
        DataInfo "input1" : tensor<1x16x32x17xf16>
        DataInfo "input2" : tensor<1x16x32x17xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x16x32x18xf16>
        DataInfo "output1" : tensor<1x16x32x19xf16>
        DataInfo "output2" : tensor<1x16x32x18xf16>
    }

func.func @main(%arg0:  memref<1x16x32x17xf16>, %arg1:  memref<1x16x32x17xf16>, %arg2:  memref<1x16x32x17xf16>, %arg3:  memref<1x16x32x18xf16, #NHWC, [@CMX_NN, 0]>, %arg4:  memref<1x16x32x19xf16, #NHWC, [@CMX_NN, 1]>, %arg5:  memref<1x16x32x18xf16, #NHWC, [@CMX_NN, 2]>) -> (memref<1x16x32x18xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x32x19xf16, #NHWC, [@CMX_NN, 1]>, memref<1x16x32x18xf16, #NHWC, [@CMX_NN, 2]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <18432> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <18432> ->  !OutputITI2

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <37888> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <37888> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <37888> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1 : !OutputITI1)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over H
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [16, 15, 15],
                    inStart = [0, 0, 0],
                    inEnd = [16, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 16, 0],
                    outEnd = [16, 31, 15],
                    inStart = [0, 16, 0],
                    inEnd = [16, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2: !OutputITI0, !OutputITI2)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over W
                DPUTask {
                    outStart = [1, 0, 0],
                    outEnd = [9, 31, 15],
                    inStart = [0, 0, 0],
                    inEnd = [8, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [10, 0, 0],
                    outEnd = [17, 31, 15],
                    inStart = [9, 0, 0],
                    inEnd = [16, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output1: !OutputITI1)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over W
                DPUTask {
                    outStart = [1, 0, 0],
                    outEnd = [9, 31, 15],
                    inStart = [0, 0, 0],
                    inEnd = [8, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [10, 0, 0],
                    outEnd = [17, 31, 15],
                    inStart = [9, 0, 0],
                    inEnd = [16, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    return %arg3, %arg4, %arg5: memref<1x16x32x18xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x32x19xf16, #NHWC, [@CMX_NN, 1]>, memref<1x16x32x18xf16, #NHWC, [@CMX_NN, 2]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:      variants : {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 16 : i64, xEnd = 16 : i64, yStart = 0 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -512 : i64, targetClusters = [1], targetWidth = 19 : i64>],
// CHECK-SAME:               inEnd = [16, 15, 15],
// CHECK-SAME:               inStart = [0, 0, 0],
// CHECK-SAME:               outEnd = [16, 15, 15],
// CHECK-SAME:               outStart = [0, 0, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 16 : i64, xEnd = 16 : i64, yStart = 16 : i64, yEnd = 31 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -512 : i64, targetClusters = [1], targetWidth = 19 : i64>],
// CHECK-SAME:               inEnd = [16, 31, 15],
// CHECK-SAME:               inStart = [0, 16, 0],
// CHECK-SAME:               outEnd = [16, 31, 15],
// CHECK-SAME:               outStart = [0, 16, 0],

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 0 : i64, yEnd = 31 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 512 : i64, targetClusters = [0], targetWidth = 18 : i64>],
// CHECK-SAME:               inEnd = [8, 31, 15],
// CHECK-SAME:               inStart = [0, 0, 0],
// CHECK-SAME:               outEnd = [9, 31, 15],
// CHECK-SAME:               outStart = [1, 0, 0],
//CHECK:                DPUTask {
//CHECK-SAME:                cluster_id = 1 : i64,
//CHECK-SAME:                haloRegions = [
//CHECK-SAME:                    #VPUIP.DPUHaloRegionAttr<xStart = 17 : i64, xEnd = 17 : i64, yStart = 0 : i64, yEnd = 31 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -544 : i64, targetClusters = [2], targetWidth = 18 : i64>],
//CHECK-SAME:                inEnd = [16, 31, 15],
//CHECK-SAME:                inStart = [9, 0, 0],
//CHECK-SAME:                outEnd = [17, 31, 15],
//CHECK-SAME:                outStart = [10, 0, 0],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 0 : i64, yEnd = 31 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 544 : i64, targetClusters = [1], targetWidth = 19 : i64>],
//CHECK-SAME:               inEnd = [8, 31, 15],
//CHECK-SAME:               inStart = [0, 0, 0],
//CHECK-SAME:               outEnd = [9, 31, 15],
//CHECK-SAME:               outStart = [1, 0, 0],
//CHECK:               DPUTask {
//CHECK-SAME:               haloRegions = [],
//CHECK-SAME:               outEnd = [17, 31, 15],
//CHECK-SAME:               outStart = [10, 0, 0],

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 1]
>

!Input2 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 2]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NHWC, [@CMX_NN, 0], // Front slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2>
                ]>
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NHWC, [@CMX_NN, 1],  // Middle slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 1, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 1, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2>
                ]>
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NHWC, [@CMX_NN, 2], // Back slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1>
                ]>
]>

// CHECK-LABEL: @ThreeNCEClusterTasksSOK
module @ThreeNCEClusterTasksSOK {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x10x10xf16>
        DataInfo "input1" : tensor<1x16x10x10xf16>
        DataInfo "input2" : tensor<1x16x10x10xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x96x10x10xf16>
        DataInfo "output1" : tensor<1x96x10x10xf16>
        DataInfo "output2" : tensor<1x96x10x10xf16>
    }

func.func @main(%arg0:  memref<1x16x10x10xf16>, %arg1:  memref<1x16x10x10xf16>, %arg2:  memref<1x16x10x10xf16>, %arg3:  memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 0]>, %arg4:  memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 1]>, %arg5:  memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 2]>) -> (memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 1]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 2]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <3200> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <3200> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <3200> ->  !OutputITI2

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1, %output2 : !OutputITI1, !OutputITI2)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [9, 9, 15],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 0, 16],
                    outEnd = [9, 9, 31],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2: !OutputITI0, !OutputITI2)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 32],
                    outEnd = [9, 9, 47],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [0, 0, 48],
                    outEnd = [9, 9, 63],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output0, %output1: !OutputITI0, !OutputITI1)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 64],
                    outEnd = [9, 9, 79],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [0, 0, 80],
                    outEnd = [9, 9, 95],
                    inStart = [9, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    return %arg3, %arg4, %arg5: memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 1]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 2]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 10 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 15],
// CHECK-SAME:               outStart = [0, 0, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 10 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 31],
// CHECK-SAME:               outStart = [0, 0, 16],

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 32 : i64, zEnd = 47 : i64, targetOffset = 0 : i64, targetClusters = [0, 2], targetWidth = 10 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 47],
// CHECK-SAME:               outStart = [0, 0, 32],
//CHECK:                DPUTask {
//CHECK-SAME:                cluster_id = 1 : i64,
//CHECK-SAME:                haloRegions = [
//CHECK-SAME:                    #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 48 : i64, zEnd = 63 : i64, targetOffset = 0 : i64, targetClusters = [0, 2], targetWidth = 10 : i64>],
//CHECK-SAME:                outEnd = [9, 9, 63],
//CHECK-SAME:                outStart = [0, 0, 48],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 64 : i64, zEnd = 79 : i64, targetOffset = 0 : i64, targetClusters = [0, 1], targetWidth = 10 : i64>],
//CHECK-SAME:               outEnd = [9, 9, 79],
//CHECK-SAME:               outStart = [0, 0, 64],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 80 : i64, zEnd = 95 : i64, targetOffset = 0 : i64, targetClusters = [0, 1], targetWidth = 10 : i64>],
//CHECK-SAME:               outEnd = [9, 9, 95],
//CHECK-SAME:               outStart = [0, 0, 80],

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x16x16xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x16x16xf16, #NHWC, [@CMX_NN, 1]
>

!Input2 = memref<
    1x16x16x16xf16, #NHWC, [@CMX_NN, 2]
>

!Input3 = memref<
    1x16x16x16xf16, #NHWC, [@CMX_NN, 3]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 0], // top height half, first half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 8, 0], cluster_id = 0>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 0>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 0>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 0>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 0> // from Tile 3
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 0, 0, 0], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 16, 0, 0], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 16, 0, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 14, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 3>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 14, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 2>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 3>
                ]>
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 1], // top height half, second half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 16, 0, 0], cluster_id = 1>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 1>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 1>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 1>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 1> // from Tile 3
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 8, 0], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 8, 0], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 14, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 2>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 3>
                ]>
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 2], // bottom height half, first half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 0], cluster_id = 2>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 8], cluster_id = 2>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 2>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 2> // fom Tile 1
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 2, 0], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 2, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 10, 0], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 10, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 2, 0], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 2, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 10, 0], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 10, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 2, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 1>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 2, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 1>
                ]>
]>

!OutputITI3 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 3], // bottom half height, second half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 2, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 10, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 2, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 10, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 3>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 3>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 3> // from Tile 1
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 0], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 0], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 8], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 8], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 2, 0], cluster_id = 3, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 1>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 2, 8], cluster_id = 3, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 1>
                ]>
]>

// CHECK-LABEL: @FourNCEClusterTasksSOHK
module @FourNCEClusterTasksSOHK {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x16x16xf16>
        DataInfo "input1" : tensor<1x16x16x16xf16>
        DataInfo "input2" : tensor<1x16x16x16xf16>
        DataInfo "input3" : tensor<1x16x16x16xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x64x18x16xf16>
        DataInfo "output1" : tensor<1x64x18x16xf16>
        DataInfo "output2" : tensor<1x64x18x16xf16>
        DataInfo "output3" : tensor<1x64x18x16xf16>
    }

func.func @main(%arg0:  memref<1x16x16x16xf16>, %arg1:  memref<1x16x16x16xf16>, %arg2:  memref<1x16x16x16xf16>, %arg3:  memref<1x16x16x16xf16>, %arg4:  memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 0]>, %arg5:  memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 1]>, %arg6:  memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 2]>, %arg7:  memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 3]>) -> (memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 1]>, memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 2]>, memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 3]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2
    %input3 = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> !Input3

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <8192> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <8192> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <8192> ->  !OutputITI2
    %output3 = VPURT.DeclareBuffer <CMX_NN> [3] <8192> ->  !OutputITI3

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <45056> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <45056> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <45056> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>
    %weights3 = VPURT.DeclareBuffer <CMX_NN> [3] <45056> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 3]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1, %output2, %output3 : !OutputITI1, !OutputITI2, !OutputITI3)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads spilt over K
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [15, 15, 15],
                    inStart = [0, 0, 0],
                    inEnd = [15, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 0, 16],
                    outEnd = [15, 15, 31],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2, %output3: !OutputITI0, !OutputITI2, !OutputITI3)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over H
                DPUTask {
                    outStart = [0, 0, 32],
                    outEnd = [15, 7, 63],
                    inStart = [0, 0, 0],
                    inEnd = [15, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [0, 8, 32],
                    outEnd = [15, 15, 63],
                    inStart = [0, 8, 0],
                    inEnd = [15, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output0, %output1, %output3: !OutputITI0, !OutputITI1, !OutputITI3)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over H & K
                DPUTask {
                    outStart = [0, 2, 0],
                    outEnd = [15, 9, 15],
                    inStart = [0, 0, 0],
                    inEnd = [15, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [0, 10, 0],
                    outEnd = [15, 17, 15],
                    inStart = [0, 8, 0],
                    inEnd = [15, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [0, 2, 16],
                    outEnd = [15, 9, 31],
                    inStart = [0, 0, 0],
                    inEnd = [15, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [0, 10, 16],
                    outEnd = [15, 17, 31],
                    inStart = [0, 8, 0],
                    inEnd = [15, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input3: !Input3)
            weights(%weights3: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 3]>)
            parent_input(%input3: !Input3)
            parent_output(%output3: !OutputITI3)
            output_ITI_buff(%output0, %output1, %output2: !OutputITI0, !OutputITI1, !OutputITI2)
            outputs(%output3: !OutputITI3)
            -> !OutputITI3
            variants : { // Workloads split over W
                DPUTask {
                    outStart = [0, 2, 32],
                    outEnd = [7, 17, 63],
                    inStart = [0, 0, 0],
                    inEnd = [7, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 3
                }
                DPUTask {
                    outStart = [8, 2, 32],
                    outEnd = [15, 17, 63],
                    inStart = [8, 0, 0],
                    inEnd = [15, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 3
                }
            } PPE : {
            }
    }

    return %arg4, %arg5, %arg6, %arg7: memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 0]>, memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 1]>, memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 2]>, memref<1x64x18x16xf16, #NHWC, [@CMX_NN, 3]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 0 : i64, targetClusters = [1], targetWidth = 16 : i64>,
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 14 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -28672 : i64, targetClusters = [2, 3], targetWidth = 16 : i64>]
// CHECK-SAME:               outEnd = [15, 15, 15],
// CHECK-SAME:               outStart = [0, 0, 0],

// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 15 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = 0 : i64, targetClusters = [1], targetWidth = 16 : i64>,
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 14 : i64, yEnd = 15 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = -28672 : i64, targetClusters = [2, 3], targetWidth = 16 : i64>
// CHECK-SAME:               outEnd = [15, 15, 31],
// CHECK-SAME:               outStart = [0, 0, 16],


//CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 7 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = 0 : i64, targetClusters = [0], targetWidth = 16 : i64>
// CHECK-SAME:               outEnd = [15, 7, 63],
// CHECK-SAME:               outStart = [0, 0, 32],
//CHECK:                DPUTask {
//CHECK-SAME:                cluster_id = 1 : i64,
//CHECK-SAME:                haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 8 : i64, yEnd = 15 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = 0 : i64, targetClusters = [0], targetWidth = 16 : i64>,
//CHECK-SAME:                    #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 14 : i64, yEnd = 15 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = -28672 : i64, targetClusters = [2, 3], targetWidth = 16 : i64>
//CHECK-SAME:                outEnd = [15, 15, 63],
//CHECK-SAME:                outStart = [0, 8, 32],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
// CHECK-SAME:                  #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 9 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 0 : i64, targetClusters = [3], targetWidth = 16 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 28672 : i64, targetClusters = [0, 1], targetWidth = 16 : i64>
//CHECK-SAME:               outEnd = [15, 9, 15],
//CHECK-SAME:               outStart = [0, 2, 0],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 10 : i64, yEnd = 17 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 0 : i64, targetClusters = [3], targetWidth = 16 : i64>
//CHECK-SAME:               outEnd = [15, 17, 15],
//CHECK-SAME:               outStart = [0, 10, 0],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 9 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = 0 : i64, targetClusters = [3], targetWidth = 16 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = 28672 : i64, targetClusters = [0, 1], targetWidth = 16 : i64>
//CHECK-SAME:               outEnd = [15, 9, 31],
//CHECK-SAME:               outStart = [0, 2, 16],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 10 : i64, yEnd = 17 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = 0 : i64, targetClusters = [3], targetWidth = 16 : i64>
//CHECK-SAME:               outEnd = [15, 17, 31],
//CHECK-SAME:               outStart = [0, 10, 16],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 3 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 7 : i64, yStart = 2 : i64, yEnd = 17 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = 0 : i64, targetClusters = [2], targetWidth = 16 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 7 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = 28672 : i64, targetClusters = [0, 1], targetWidth = 16 : i64>
//CHECK-SAME:               outEnd = [7, 17, 63],
//CHECK-SAME:               outStart = [0, 2, 32],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 3 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 8 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 17 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = 0 : i64, targetClusters = [2], targetWidth = 16 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 8 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 32 : i64, zEnd = 63 : i64, targetOffset = 28672 : i64, targetClusters = [0, 1], targetWidth = 16 : i64>
//CHECK-SAME:               outEnd = [15, 17, 63],
//CHECK-SAME:               outStart = [8, 2, 32],

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x20x16xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x20x16xf16, #NHWC, [@CMX_NN, 1]
>

!Input2 = memref<
    1x16x20x16xf16, #NHWC, [@CMX_NN, 2]
>

!Input3 = memref<
    1x16x20x16xf16, #NHWC, [@CMX_NN, 3]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 0], // top - left
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 16], cluster_id = 0>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 0], cluster_id = 0>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 16], cluster_id = 0> // from Tile 3
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 0, 15], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 0, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 10, 15], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 10, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 18, 0], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 18, 15], cluster_id = 0, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 0], cluster_id = 3>]>
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 1], // top - right
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 0, 0], cluster_id = 1>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 10, 0], cluster_id = 1>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 1], cluster_id = 1>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 9], cluster_id = 1>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 0], cluster_id = 1> // from Tile 2
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 1], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 16], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 9], offset = [0, 0, 18, 1], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 9], offset = [0, 0, 0, 1], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 7], offset = [0, 0, 18, 10], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 7], offset = [0, 0, 0, 10], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 18, 1], cluster_id = 1, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 16], cluster_id = 2>]>
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 2], // bottom - left
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 2, 16], cluster_id = 2>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 12, 16], cluster_id = 2>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 16], cluster_id = 2> // from Tile 1
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 15], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 2, 0], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 0], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 2, 15], cluster_id = 2, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 0], cluster_id = 1>]>
]>

!OutputITI3 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 3], // bottom - right
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 9], offset = [0, 0, 0, 1], cluster_id = 3>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 7], offset = [0, 0, 0, 10], cluster_id = 3>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 0], cluster_id = 3> // from Tile 0
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 2, 1], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 2, 16], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 12, 1], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 12, 16], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 2, 1], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 1], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 2, 9], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 9], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 2, 1], cluster_id = 3, inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 16], cluster_id = 0>]>
]>

// CHECK-LABEL: @FourNCEClusterTasksSOHW
module @FourNCEClusterTasksSOHW {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x20x16xf16>
        DataInfo "input1" : tensor<1x16x20x16xf16>
        DataInfo "input2" : tensor<1x16x20x16xf16>
        DataInfo "input3" : tensor<1x16x20x16xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x16x22x17xf16>
        DataInfo "output1" : tensor<1x16x22x17xf16>
        DataInfo "output2" : tensor<1x16x22x17xf16>
        DataInfo "output3" : tensor<1x16x22x17xf16>
    }

func.func @main(%arg0:  memref<1x16x20x16xf16>, %arg1:  memref<1x16x20x16xf16>, %arg2:  memref<1x16x20x16xf16>, %arg3:  memref<1x16x20x16xf16>, %arg4:  memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 0]>, %arg5:  memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 1]>, %arg6:  memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 2]>, %arg7:  memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 3]>) -> (memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 1]>, memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 2]>, memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 3]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2
    %input3 = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> !Input3

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <10880> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <10880> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <10880> ->  !OutputITI2
    %output3 = VPURT.DeclareBuffer <CMX_NN> [3] <10880> ->  !OutputITI3

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <22848> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <22848> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <22848> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>
    %weights3 = VPURT.DeclareBuffer <CMX_NN> [3] <22848> -> memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 3]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1, %output2, %output3 : !OutputITI1, !OutputITI2, !OutputITI3)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads spilt over H
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [15, 9, 15],
                    inStart = [0, 0, 0],
                    inEnd = [15, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 10, 0],
                    outEnd = [15, 19, 15],
                    inStart = [0, 0, 0],
                    inEnd = [15, 19, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2, %output3: !OutputITI0, !OutputITI2, !OutputITI3)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over W
                DPUTask {
                    outStart = [1, 0, 0],
                    outEnd = [9, 19, 15],
                    inStart = [0, 0, 0],
                    inEnd = [8, 19, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [10, 0, 0],
                    outEnd = [16, 19, 15],
                    inStart = [9, 0, 0],
                    inEnd = [15, 19, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output0, %output1, %output3: !OutputITI0, !OutputITI1, !OutputITI3)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // No workload split
                DPUTask {
                    outStart = [0, 2, 0],
                    outEnd = [15, 21, 15],
                    inStart = [0, 0, 0],
                    inEnd = [15, 19, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input3: !Input3)
            weights(%weights3: memref<16x16x1x1xf16, #NHWC, [@CMX_NN, 3]>)
            parent_input(%input3: !Input3)
            parent_output(%output3: !OutputITI3)
            output_ITI_buff(%output0, %output1, %output2: !OutputITI0, !OutputITI1, !OutputITI2)
            outputs(%output3: !OutputITI3)
            -> !OutputITI3
            variants : { // Workloads split over H & W
                DPUTask {
                    outStart = [1, 2, 0],
                    outEnd = [8, 11, 15],
                    inStart = [0, 0, 0],
                    inEnd = [7, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 3
                }
                DPUTask {
                    outStart = [9, 2, 0],
                    outEnd = [16, 11, 15],
                    inStart = [8, 0, 0],
                    inEnd = [15, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 3
                }
                DPUTask {
                    outStart = [1, 12, 0],
                    outEnd = [8, 21, 15],
                    inStart = [0, 10, 0],
                    inEnd = [7, 19, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 3
                }
                DPUTask {
                    outStart = [9, 12, 0],
                    outEnd = [16, 21, 15],
                    inStart = [8, 10, 0],
                    inEnd = [15, 19, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 3
                }
            } PPE : {
            }
    }

    return %arg4, %arg5, %arg6, %arg7: memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 1]>, memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 2]>, memref<1x16x22x17xf16, #NHWC, [@CMX_NN, 3]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 15 : i64, xEnd = 15 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -480 : i64, targetClusters = [1], targetWidth = 17 : i64>],
// CHECK-SAME:               outEnd = [15, 9, 15],
// CHECK-SAME:               outStart = [0, 0, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 15 : i64, xEnd = 15 : i64, yStart = 10 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -480 : i64, targetClusters = [1], targetWidth = 17 : i64>,
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 18 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -9792 : i64, targetClusters = [2], targetWidth = 17 : i64>,
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 15 : i64, xEnd = 15 : i64, yStart = 18 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -10272 : i64, targetClusters = [3], targetWidth = 17 : i64>],
// CHECK-SAME:               outEnd = [15, 19, 15],
// CHECK-SAME:               outStart = [0, 10, 0],

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 0 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 480 : i64, targetClusters = [0], targetWidth = 17 : i64>,
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 9 : i64, yStart = 18 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -9792 : i64, targetClusters = [3], targetWidth = 17 : i64>,
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 18 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -9312 : i64, targetClusters = [2], targetWidth = 17 : i64>],
// CHECK-SAME:               outEnd = [9, 19, 15],
// CHECK-SAME:               outStart = [1, 0, 0],
//CHECK:                DPUTask {
//CHECK-SAME:                cluster_id = 1 : i64,
//CHECK-SAME:                haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 10 : i64, xEnd = 16 : i64, yStart = 18 : i64, yEnd = 19 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -9792 : i64, targetClusters = [3], targetWidth = 17 : i64>
//CHECK-SAME:                outEnd = [16, 19, 15],
//CHECK-SAME:                outStart = [10, 0, 0],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 15 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 21 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = -480 : i64, targetClusters = [3], targetWidth = 17 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 9792 : i64, targetClusters = [0], targetWidth = 17 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 15 : i64, xEnd = 15 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 9312 : i64, targetClusters = [1], targetWidth = 17 : i64>
//CHECK-SAME:               outEnd = [15, 21, 15],
//CHECK-SAME:               outStart = [0, 2, 0],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 3 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 2 : i64, yEnd = 11 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 480 : i64, targetClusters = [2], targetWidth = 17 : i64>,
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 8 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 9792 : i64, targetClusters = [1], targetWidth = 17 : i64>,
// CHECK-SAME:                  #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 10272 : i64, targetClusters = [0], targetWidth = 17 : i64>
//CHECK-SAME:               outEnd = [8, 11, 15],
//CHECK-SAME:               outStart = [1, 2, 0],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 3 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 9 : i64, xEnd = 16 : i64, yStart = 2 : i64, yEnd = 3 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 9792 : i64, targetClusters = [1], targetWidth = 17 : i64>
//CHECK-SAME:               outEnd = [16, 11, 15],
//CHECK-SAME:               outStart = [9, 2, 0],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 3 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 1 : i64, xEnd = 1 : i64, yStart = 12 : i64, yEnd = 21 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 480 : i64, targetClusters = [2], targetWidth = 17 : i64>
//CHECK-SAME:               outEnd = [8, 21, 15],
//CHECK-SAME:               outStart = [1, 12, 0],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 3 : i64,
//CHECK-SAME:               haloRegions = [],
//CHECK-SAME:               outEnd = [16, 21, 15],
//CHECK-SAME:               outStart = [9, 12, 0],


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 1]
>

!Input2 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 2]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NCHW, [@CMX_NN, 0], // Front slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2>
                ]>
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NCHW, [@CMX_NN, 1],  // Middle slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 1, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 1, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2>
                ]>
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NCHW, [@CMX_NN, 2], // Back slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1>
                ]>
]>

// CHECK-LABEL: @SOK_ODUPermute
module @SOK_ODUPermute attributes {config.arch = #config.arch_kind<NPU40XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x10x10xf16>
        DataInfo "input1" : tensor<1x16x10x10xf16>
        DataInfo "input2" : tensor<1x16x10x10xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x96x10x10xf16>
        DataInfo "output1" : tensor<1x96x10x10xf16>
        DataInfo "output2" : tensor<1x96x10x10xf16>
    }

func.func @main(%arg0:  memref<1x16x10x10xf16>, %arg1:  memref<1x16x10x10xf16>, %arg2:  memref<1x16x10x10xf16>, %arg3:  memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 0]>, %arg4:  memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 1]>, %arg5:  memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 2]>) -> (memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 1]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 2]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <3200> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <3200> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <3200> ->  !OutputITI2

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1, %output2 : !OutputITI1, !OutputITI2)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [9, 9, 15],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 0, 16],
                    outEnd = [9, 9, 31],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2: !OutputITI0, !OutputITI2)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 32],
                    outEnd = [9, 9, 47],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [0, 0, 48],
                    outEnd = [9, 9, 63],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output0, %output1: !OutputITI0, !OutputITI1)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 64],
                    outEnd = [9, 9, 79],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [0, 0, 80],
                    outEnd = [9, 9, 95],
                    inStart = [9, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    return %arg3, %arg4, %arg5: memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 0]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 1]>, memref<1x96x10x10xf16, #NHWC, [@CMX_NN, 2]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 9 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 10 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 15],
// CHECK-SAME:               outStart = [0, 0, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 16 : i64, yEnd = 31 : i64, zStart = 0 : i64, zEnd = 9 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 10 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 31],
// CHECK-SAME:               outStart = [0, 0, 16],

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 32 : i64, yEnd = 47 : i64, zStart = 0 : i64, zEnd = 9 : i64, targetOffset = 0 : i64, targetClusters = [0, 2], targetWidth = 10 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 47],
// CHECK-SAME:               outStart = [0, 0, 32],
//CHECK:                DPUTask {
//CHECK-SAME:                cluster_id = 1 : i64,
//CHECK-SAME:                haloRegions = [
//CHECK-SAME:                    #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 48 : i64, yEnd = 63 : i64, zStart = 0 : i64, zEnd = 9 : i64, targetOffset = 0 : i64, targetClusters = [0, 2], targetWidth = 10 : i64>],
//CHECK-SAME:                outEnd = [9, 9, 63],
//CHECK-SAME:                outStart = [0, 0, 48],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 64 : i64, yEnd = 79 : i64, zStart = 0 : i64, zEnd = 9 : i64, targetOffset = 0 : i64, targetClusters = [0, 1], targetWidth = 10 : i64>],
//CHECK-SAME:               outEnd = [9, 9, 79],
//CHECK-SAME:               outStart = [0, 0, 64],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 80 : i64, yEnd = 95 : i64, zStart = 0 : i64, zEnd = 9 : i64, targetOffset = 0 : i64, targetClusters = [0, 1], targetWidth = 10 : i64>],
//CHECK-SAME:               outEnd = [9, 9, 95],
//CHECK-SAME:               outStart = [0, 0, 80],


// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 1]
>

!Input2 = memref<
    1x16x10x10xf16, #NHWC, [@CMX_NN, 2]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 0], // Front slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 0, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2>
                ]>
]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 1],  // Middle slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 1, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 1, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2>
                ]>
]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 2], // Back slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>
                ]>,
        #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 2, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1>
                ]>
]>

// CHECK-LABEL: @ThreeNCEClusterTasksSOKWithHStride
module @ThreeNCEClusterTasksSOKWithHStride {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x10x10xf16>
        DataInfo "input1" : tensor<1x16x10x10xf16>
        DataInfo "input2" : tensor<1x16x10x10xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x96x10x10xf16>
        DataInfo "output1" : tensor<1x96x10x10xf16>
        DataInfo "output2" : tensor<1x96x10x10xf16>
    }

func.func @main(%arg0:  memref<1x16x10x10xf16>, %arg1:  memref<1x16x10x10xf16>, %arg2:  memref<1x16x10x10xf16>, %arg3:  memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 0]>, %arg4:  memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 1]>, %arg5:  memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 2]>) -> (memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 0]>, memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 1]>, memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 2]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <3200> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <3200> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <3200> ->  !OutputITI2

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <22400> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1, %output2 : !OutputITI1, !OutputITI2)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [9, 9, 15],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 0, 16],
                    outEnd = [9, 9, 31],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2: !OutputITI0, !OutputITI2)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 32],
                    outEnd = [9, 9, 47],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [0, 0, 48],
                    outEnd = [9, 9, 63],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output0, %output1: !OutputITI0, !OutputITI1)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 0, 64],
                    outEnd = [9, 9, 79],
                    inStart = [0, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [0, 0, 80],
                    outEnd = [9, 9, 95],
                    inStart = [9, 0, 0],
                    inEnd = [9, 9, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    return %arg3, %arg4, %arg5: memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 0]>, memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 1]>, memref<1x96x10x10xf16, {order = #NHWC, strides = [19200, 1, 1920, 96]}, [@CMX_NN, 2]>
}

}

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 0 : i64, zEnd = 15 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 20 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 15],
// CHECK-SAME:               outStart = [0, 0, 0],
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 0 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 16 : i64, zEnd = 31 : i64, targetOffset = 0 : i64, targetClusters = [1, 2], targetWidth = 20 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 31],
// CHECK-SAME:               outStart = [0, 0, 16],

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:               DPUTask {
// CHECK-SAME:               cluster_id = 1 : i64,
// CHECK-SAME:               haloRegions = [
// CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 32 : i64, zEnd = 47 : i64, targetOffset = 0 : i64, targetClusters = [0, 2], targetWidth = 20 : i64>],
// CHECK-SAME:               outEnd = [9, 9, 47],
// CHECK-SAME:               outStart = [0, 0, 32],
//CHECK:                DPUTask {
//CHECK-SAME:                cluster_id = 1 : i64,
//CHECK-SAME:                haloRegions = [
//CHECK-SAME:                    #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 48 : i64, zEnd = 63 : i64, targetOffset = 0 : i64, targetClusters = [0, 2], targetWidth = 20 : i64>],
//CHECK-SAME:                outEnd = [9, 9, 63],
//CHECK-SAME:                outStart = [0, 0, 48],

//CHECK:         VPUIP.NCEClusterTask {
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 64 : i64, zEnd = 79 : i64, targetOffset = 0 : i64, targetClusters = [0, 1], targetWidth = 20 : i64>],
//CHECK-SAME:               outEnd = [9, 9, 79],
//CHECK-SAME:               outStart = [0, 0, 64],
//CHECK:               DPUTask {
//CHECK-SAME:               cluster_id = 2 : i64,
//CHECK-SAME:               haloRegions = [
//CHECK-SAME:                   #VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 9 : i64, yStart = 0 : i64, yEnd = 9 : i64, zStart = 80 : i64, zEnd = 95 : i64, targetOffset = 0 : i64, targetClusters = [0, 1], targetWidth = 20 : i64>],
//CHECK-SAME:               outEnd = [9, 9, 95],
//CHECK-SAME:               outStart = [0, 0, 80],
