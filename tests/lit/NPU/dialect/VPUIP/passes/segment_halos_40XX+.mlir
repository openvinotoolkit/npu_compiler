//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --segment-halos %s | FileCheck %s
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
    1x32x17x32xf16, #NHWC, [@CMX_NN, 0], // top half of height
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 1, 32], offset = [0, 0, 15, 0], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>]>]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x32x17x32xf16, #NHWC, [@CMX_NN, 1], // bottom half of height
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 1, 32], offset = [0, 0, 1, 0], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>]>]>

!OutputITISparse0 = !VPUIP.ITIBuffer<
    1x32x17x32xi1, #NHWC, [@CMX_NN, 0], // top half of height
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 1, 32], offset = [0, 0, 15, 0], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>]>]>

!OutputITISparse1 = !VPUIP.ITIBuffer<
    1x32x17x32xi1, #NHWC, [@CMX_NN, 1], // bottom half of height
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1>],
    outwardHaloRegions = [#VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 1, 32], offset = [0, 0, 1, 0], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0>]>]>

// CHECK-LABEL: @TwoNCEClusterTasksSOH
module @TwoNCEClusterTasksSOH {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x17x32xf16>
        DataInfo "input1" : tensor<1x16x17x32xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x32x17x32xf16>
        DataInfo "output1" : tensor<1x32x17x32xf16>
    }

func.func @main(%arg0:  memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 0]>, %arg1:  memref<1x16x17x32xf16, #NHWC, [@CMX_NN, 1]>, %arg2:  memref<1x32x17x32xf16, #NHWC, [@CMX_NN, 0]>, %arg3:  memref<1x32x17x32xf16, #NHWC, [@CMX_NN, 1]>) -> (memref<1x32x17x32xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x17x32xf16, #NHWC, [@CMX_NN, 1]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> ->  !OutputITI1

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>

    %output_sm0 = VPURT.DeclareBuffer <CMX_NN> [0] <39680> -> !OutputITISparse0
    %output_sm1 = VPURT.DeclareBuffer <CMX_NN> [1] <39680> -> !OutputITISparse1

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
            parent_output_sparsity_map(%output_sm0 : !OutputITISparse0)
            output_ITI_buff(%output1 : !OutputITI1)
            outputs(%output0: !OutputITI0)
            output_sparsity_map(%output_sm0 : !OutputITISparse0)
            -> !OutputITI0, !OutputITISparse0
            variants : { // Workloads split over H
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [31, 7, 31],
                    inStart = [0, 0, 0],
                    inEnd = [31, 7, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 8, 0],
                    outEnd = [31, 15, 31],
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
            weights(%weights1: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            parent_output_sparsity_map(%output_sm1 : !OutputITISparse1)
            output_ITI_buff(%output0: !OutputITI0)
            outputs(%output1: !OutputITI1)
            output_sparsity_map(%output_sm1 : !OutputITISparse1)
            -> !OutputITI1, !OutputITISparse1
            variants : { // Workloads split over K
                DPUTask {
                    outStart = [0, 1, 0],
                    outEnd = [31, 16, 15],
                    inStart = [0, 0, 0],
                    inEnd = [31, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [0, 1, 16],
                    outEnd = [31, 16, 31],
                    inStart = [0, 0, 0],
                    inEnd = [31, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
            } PPE : {
            }
    }

    return %arg2, %arg3: memref<1x32x17x32xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x17x32xf16, #NHWC, [@CMX_NN, 1]>
}

}

// CHECK:       [[OUT_CMX0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x17x32xf16, #NHWC, [@CMX_NN, 0],
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 16, 16, 0], cluster_id = 0 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:               #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:              shape = [1, 32, 1, 32],
// CHECK-SAME:              offset = [0, 0, 15, 0],
// CHECK-SAME:              cluster_id = 0 : i64,
// CHECK-SAME:              inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:              ]>
// CHECK:           ]>

// CHECK:       [[OUT_CMX1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x17x32xf16, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:              #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 1, 32],
// CHECK-SAME:           offset = [0, 0, 1, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 1, 32],
// CHECK-SAME:           offset = [0, 16, 1, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 16, 16, 0], cluster_id = 0 : i64>
// CHECK-NEXT:           ]>

// CHECK:       [[OUT_SPARSE_CMX0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <39680> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x17x32xi1, #NHWC, [@CMX_NN, 0]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 16, 16, 0], cluster_id = 0 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 1, 32],
// CHECK-SAME:           offset = [0, 0, 15, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:           ]>

// CHECK:       [[OUT_SPARSE_CMX1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <39680> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x17x32xi1, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 32, 1, 32], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 1, 32],
// CHECK-SAME:           offset = [0, 0, 1, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 0, 16, 0], cluster_id = 0 : i64>
// CHECK-NEXT:      ]>,
// CHECK-NEXT:      #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 1, 32],
// CHECK-SAME:           offset = [0, 16, 1, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 1, 32], offset = [0, 16, 16, 0], cluster_id = 0 : i64>
// CHECK:           ]>


// CHECK:        VPUIP.NCEClusterTask {
// CHECK:          parent_output_sparsity_map([[OUT_SPARSE_CMX0]]
// CHECK:          output_ITI_buff([[OUT_CMX1]]
// CHECK:          outputs([[OUT_CMX0]]
// CHECK:          output_sparsity_map([[OUT_SPARSE_CMX0]]

// CHECK:         VPUIP.NCEClusterTask {
// CHECK:           parent_output_sparsity_map([[OUT_SPARSE_CMX1]]
// CHECK:           output_ITI_buff([[OUT_CMX0]]
// CHECK:           outputs([[OUT_CMX1]]
// CHECK:           output_sparsity_map([[OUT_SPARSE_CMX1]]

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
    1x32x32x18xf16, #NHWC, [@CMX_NN, 0], // left-most slice of W
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 17], cluster_id = 0>],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 32, 1], offset = [0, 0, 0, 16], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 0], cluster_id = 1>]>
    ]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x32x32x19xf16, #NHWC, [@CMX_NN, 1], // middle slice of W
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 18], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 32, 1], offset = [0, 0, 0, 1], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 17], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 32, 1], offset = [0, 0, 0, 17], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 0], cluster_id = 2>]>
    ]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x32x32x18xf16, #NHWC, [@CMX_NN, 2], // right-most slice of W
    inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 0], cluster_id = 2>],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 32, 1], offset = [0, 0, 0, 1], cluster_id = 2,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 18], cluster_id = 1>]>
    ]>

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
        DataInfo "output0" : tensor<1x32x32x18xf16>
        DataInfo "output1" : tensor<1x32x32x19xf16>
        DataInfo "output2" : tensor<1x32x32x18xf16>
    }

func.func @main(%arg0:  memref<1x16x32x17xf16>, %arg1:  memref<1x16x32x17xf16>, %arg2:  memref<1x16x32x17xf16>, %arg3:  memref<1x32x32x18xf16, #NHWC, [@CMX_NN, 0]>, %arg4:  memref<1x32x32x19xf16, #NHWC, [@CMX_NN, 1]>, %arg5:  memref<1x32x32x18xf16, #NHWC, [@CMX_NN, 2]>) -> (memref<1x32x32x18xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x32x19xf16, #NHWC, [@CMX_NN, 1]>, memref<1x32x32x18xf16, #NHWC, [@CMX_NN, 2]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <18432> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <18432> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <18432> ->  !OutputITI2

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <37888> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <37888> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <37888> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>

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
            output_ITI_buff(%output1 : !OutputITI1)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over H
                DPUTask {
                    outStart = [0, 0, 0],
                    outEnd = [16, 15, 31],
                    inStart = [0, 0, 0],
                    inEnd = [16, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 0
                }
                DPUTask {
                    outStart = [0, 16, 0],
                    outEnd = [16, 31, 31],
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
            weights(%weights1: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2: !OutputITI0, !OutputITI2)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over W
                DPUTask {
                    outStart = [1, 0, 0],
                    outEnd = [9, 31, 31],
                    inStart = [0, 0, 0],
                    inEnd = [8, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 1
                }
                DPUTask {
                    outStart = [10, 0, 0],
                    outEnd = [17, 31, 31],
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
            weights(%weights2: memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output1: !OutputITI1)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over H & K
                DPUTask {
                    outStart = [1, 0, 0],
                    outEnd = [17, 15, 15],
                    inStart = [0, 0, 0],
                    inEnd = [16, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [1, 0, 16],
                    outEnd = [17, 15, 31],
                    inStart = [0, 0, 0],
                    inEnd = [16, 15, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [1, 16, 0],
                    outEnd = [17, 31, 15],
                    inStart = [0, 16, 0],
                    inEnd = [16, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
                DPUTask {
                    outStart = [1, 16, 16],
                    outEnd = [17, 31, 31],
                    inStart = [0, 16, 0],
                    inEnd = [16, 31, 15],
                    pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                    mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                    cluster_id = 2
                }
            } PPE : {
            }
    }

    return %arg3, %arg4, %arg5: memref<1x32x32x18xf16, #NHWC, [@CMX_NN, 0]>, memref<1x32x32x19xf16, #NHWC, [@CMX_NN, 1]>, memref<1x32x32x18xf16, #NHWC, [@CMX_NN, 2]>
}

}

// CHECK:       [[OUT_CMX0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <18432> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x32x18xf16, #NHWC, [@CMX_NN, 0]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 17], cluster_id = 0 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:                #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 16, 1],
// CHECK-SAME:           offset = [0, 0, 0, 16],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 16, 1],
// CHECK-SAME:           offset = [0, 0, 16, 16],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:                 #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 1], offset = [0, 0, 16, 0], cluster_id = 1 : i64>
// CHECK-NEXT:           ]>

// CHECK:       [[OUT_CMX1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <18432> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x32x19xf16, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:              #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 1], offset = [0, 0, 16, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 0, 18], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 16, 0, 18], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 16, 18], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 16, 16, 18], cluster_id = 1 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:                #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 32, 1],
// CHECK-SAME:           offset = [0, 0, 0, 1],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 17], cluster_id = 0 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 32, 1],
// CHECK-SAME:           offset = [0, 0, 0, 17],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>

// CHECK:       [[OUT_CMX2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <18432> -> !VPUIP.ITIBuffer<
// CHECK:           1x32x32x18xf16, #NHWC, [@CMX_NN, 2]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 32, 32, 1], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:               #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 16, 1],
// CHECK-SAME:           offset = [0, 0, 0, 1],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 0, 18], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 16, 1],
// CHECK-SAME:           offset = [0, 16, 0, 1],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 16, 0, 18], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 16, 1],
// CHECK-SAME:           offset = [0, 0, 16, 1],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 0, 16, 18], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 16, 1],
// CHECK-SAME:           offset = [0, 16, 16, 1],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 1], offset = [0, 16, 16, 18], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:           output_ITI_buff([[OUT_CMX1]]
// CHECK:       outputs([[OUT_CMX0]]

// CHECK:         VPUIP.NCEClusterTask {
// CHECK:           output_ITI_buff([[OUT_CMX0]], [[OUT_CMX2]]
// CHECK:           outputs([[OUT_CMX1]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:           output_ITI_buff([[OUT_CMX1]]
// CHECK:           outputs([[OUT_CMX2]]

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
        #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
        #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 10, 10], offset = [0, 0, 0, 0], cluster_id = 0,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>
                ]>
    ]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NHWC, [@CMX_NN, 1],  // Middle slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1>,
        #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 10, 10], offset = [0, 32, 0, 0], cluster_id = 1,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>
                ]>
    ]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x96x10x10xf16, #NHWC, [@CMX_NN, 2], // Back slice of K
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2>,
        #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2>
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 10, 10], offset = [0, 64, 0, 0], cluster_id = 2,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1>
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

// CHECK:       [[OUT_CMX0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <3200> -> !VPUIP.ITIBuffer<
// CHECK:           1x96x10x10xf16, #NHWC, [@CMX_NN, 0]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:               #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 10],
// CHECK-SAME:           offset = [0, 0, 0, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:           ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 10],
// CHECK-SAME:           offset = [0, 16, 0, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:           ]>

// CHECK:       [[OUT_CMX1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <3200> -> !VPUIP.ITIBuffer<
// CHECK:           1x96x10x10xf16, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:            #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 10],
// CHECK-SAME:           offset = [0, 32, 0, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 10],
// CHECK-SAME:           offset = [0, 48, 0, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>

// CHECK:       [[OUT_CMX2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <3200> -> !VPUIP.ITIBuffer<
// CHECK:           1x96x10x10xf16, #NHWC, [@CMX_NN, 2]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:              #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 16, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 32, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 48, 0, 0], cluster_id = 2 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 10],
// CHECK-SAME:           offset = [0, 64, 0, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 64, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 10],
// CHECK-SAME:           offset = [0, 80, 0, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 10], offset = [0, 80, 0, 0], cluster_id = 1 : i64
// CHECK-NEXT:          ]>

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:           output_ITI_buff([[OUT_CMX1]], [[OUT_CMX2]]
// CHECK:           outputs([[OUT_CMX0]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:           output_ITI_buff([[OUT_CMX0]], [[OUT_CMX2]]
// CHECK:           outputs([[OUT_CMX1]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:           output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]]
// CHECK:           outputs([[OUT_CMX2]]


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
        #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 32, 0, 0], cluster_id = 0>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 16, 0], cluster_id = 0>,  // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 16, 0], cluster_id = 0>  // from Tile 3
    ],

    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 16, 16], offset = [0, 0, 0, 0], cluster_id = 0,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 2, 16], offset = [0, 0, 14, 0], cluster_id = 0,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 0, 0], cluster_id = 3>]>
    ]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 1], // top height half, second half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 16, 0], cluster_id = 1>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 16, 0], cluster_id = 1> // from Tile 3
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 16, 16], offset = [0, 32, 0, 0], cluster_id = 1,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 32, 0, 0], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 2, 16], offset = [0, 32, 14, 0], cluster_id = 0,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 2>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 3>
                ]>
    ]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 2], // bottom height half, first half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 32, 2, 0], cluster_id = 2>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 2> // fom Tile 1
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 16, 16], offset = [0, 0, 2, 0], cluster_id = 2,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 0, 2, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 2, 16], offset = [0, 0, 2, 0], cluster_id = 2,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 16, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 16, 0], cluster_id = 1>
                ]>
    ]>

!OutputITI3 = !VPUIP.ITIBuffer<
    1x64x18x16xf16, #NHWC, [@CMX_NN, 3], // bottom half height, second half of channels
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 0, 2, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 0, 0, 0], cluster_id = 3>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 3> // from Tile 1
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 16, 16], offset = [0, 32, 2, 0], cluster_id = 3,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 32, 16, 16], offset = [0, 32, 2, 0], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 32, 2, 16], offset = [0, 32, 2, 0], cluster_id = 3,
                inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 16, 0], cluster_id = 0>,
                    #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 16, 0], cluster_id = 1>
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

// CHECK:       [[OUT_CMX0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <8192> -> !VPUIP.ITIBuffer<
// CHECK:           1x64x18x16xf16, #NHWC, [@CMX_NN, 0]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 8, 0], cluster_id = 0 : i64>,
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 0 : i64>
// CHECK:           ],
// CHECK:           outwardHaloRegions = [
// CHECK:             #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 16, 16],
// CHECK-SAME:           offset = [0, 0, 0, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 16, 16],
// CHECK-SAME:           offset = [0, 16, 0, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 16, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 16],
// CHECK-SAME:           offset = [0, 0, 14, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 16],
// CHECK-SAME:           offset = [0, 16, 14, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 3 : i64>
// CHECK:               ]

// CHECK:       [[OUT_CMX1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <8192> -> !VPUIP.ITIBuffer<
// CHECK:           1x64x18x16xf16, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:              #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 16, 16], offset = [0, 16, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 1 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:               #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 8, 16],
// CHECK-SAME:           offset = [0, 32, 0, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 0, 0], cluster_id = 0 : i64>
// CHECK-NEXT:           ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 8, 16],
// CHECK-SAME:           offset = [0, 32, 8, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 8, 16], offset = [0, 32, 8, 0], cluster_id = 0 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 2, 16],
// CHECK-SAME:           offset = [0, 32, 14, 0],
// CHECK-SAME:           cluster_id = 0 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 3 : i64>
// CHECK:                ]

// CHECK:       [[OUT_CMX2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <8192> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x64x18x16xf16, #NHWC, [@CMX_NN, 2]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 8], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 8, 16],
// CHECK-SAME:           offset = [0, 0, 2, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 2, 0], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 8, 16],
// CHECK-SAME:           offset = [0, 0, 10, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 10, 0], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 8, 16],
// CHECK-SAME:           offset = [0, 16, 2, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 2, 0], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 8, 16],
// CHECK-SAME:           offset = [0, 16, 10, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 10, 0], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 16],
// CHECK-SAME:           offset = [0, 0, 2, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 16, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 16],
// CHECK-SAME:           offset = [0, 16, 2, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 16, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]

// CHECK:       [[OUT_CMX3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <8192> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x64x18x16xf16, #NHWC, [@CMX_NN, 3]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 2, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 0, 10, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 2, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 8, 16], offset = [0, 16, 10, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 16, 0, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 16], offset = [0, 32, 0, 0], cluster_id = 3 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 16, 8],
// CHECK-SAME:           offset = [0, 32, 2, 0],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 0], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 16, 8],
// CHECK-SAME:           offset = [0, 32, 2, 8],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 16, 8], offset = [0, 32, 2, 8], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 2, 8],
// CHECK-SAME:           offset = [0, 32, 2, 0],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>,
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 32, 2, 8],
// CHECK-SAME:           offset = [0, 32, 2, 8],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 0 : i64>,
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 32, 2, 8], offset = [0, 32, 16, 8], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX1]], [[OUT_CMX2]], [[OUT_CMX3]]
// CHECK:     outputs([[OUT_CMX0]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX0]], [[OUT_CMX2]], [[OUT_CMX3]]
// CHECK:     outputs([[OUT_CMX1]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]], [[OUT_CMX3]]
// CHECK:     outputs([[OUT_CMX2]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]], [[OUT_CMX2]]
// CHECK:     outputs([[OUT_CMX3]]

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
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 16], cluster_id = 0>  // from Tile 3
    ],

    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 20, 1], offset = [0, 0, 0, 15], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 0], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 16], offset = [0, 0, 18, 0], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 1], offset = [0, 0, 18, 15], cluster_id = 0,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 0], cluster_id = 3>]>
    ]>

!OutputITI1 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 1], // top - right
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 0], cluster_id = 1>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 1], cluster_id = 1>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 0], cluster_id = 1> // from Tile 2
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 20, 1], offset = [0, 0, 0, 1], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 16], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 16], offset = [0, 0, 18, 1], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 1], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 1], offset = [0, 0, 18, 1], cluster_id = 1,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 16], cluster_id = 2>]>
    ]>

!OutputITI2 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 2], // bottom - left
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 16], cluster_id = 2>, // from Tile 3
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2>, // from Tile 0
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 16], cluster_id = 2> // from Tile 1
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 20, 1], offset = [0, 0, 2, 15], cluster_id = 2,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 0], cluster_id = 3>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 16], offset = [0, 0, 2, 0], cluster_id = 2,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 0], cluster_id = 0>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 1], offset = [0, 0, 2, 15], cluster_id = 2,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 0], cluster_id = 1>]>
    ]>

!OutputITI3 = !VPUIP.ITIBuffer<
    1x16x22x17xf16, #NHWC, [@CMX_NN, 3], // bottom - right
    inwardHaloRegions = [
        #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 0], cluster_id = 3>, // from Tile 2
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 1], cluster_id = 3>, // from Tile 1
        #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 0], cluster_id = 3> // from Tile 0
    ],
    outwardHaloRegions = [
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 20, 1], offset = [0, 0, 2, 1], cluster_id = 3,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 16], cluster_id = 2>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 16], offset = [0, 0, 2, 1], cluster_id = 3,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 1], cluster_id = 1>]>,
        #VPUIP.OutwardHaloRegionAttr<
            shape = [1, 16, 2, 1], offset = [0, 0, 2, 1], cluster_id = 3,
                inwardHaloRegions = [#VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 16], cluster_id = 0>]>
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

// CHECK:       [[OUT_CMX0:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <10880> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x16x22x17xf16, #NHWC, [@CMX_NN, 0]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:           #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>,
// CHECK-NEXT:           #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 0], cluster_id = 0 : i64>,
// CHECK-NEXT:           #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 16], cluster_id = 0 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:            shape = [1, 16, 10, 1],
// CHECK-SAME:            offset = [0, 0, 0, 15],
// CHECK-SAME:            cluster_id = 0 : i64,
// CHECK-SAME:            inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:            ]>
// CHECK-NEXT:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:            shape = [1, 16, 10, 1],
// CHECK-SAME:            offset = [0, 0, 10, 15],
// CHECK-SAME:            cluster_id = 0 : i64,
// CHECK-SAME:            inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 10, 0], cluster_id = 1 : i64>
// CHECK-NEXT:           ]>
// CHECK-NEXT:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:            shape = [1, 16, 2, 16],
// CHECK-SAME:            offset = [0, 0, 18, 0],
// CHECK-SAME:            cluster_id = 0 : i64,
// CHECK-SAME:            inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
// CHECK-NEXT:            ]>
// CHECK-NEXT:           #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:            shape = [1, 16, 2, 1],
// CHECK-SAME:            offset = [0, 0, 18, 15],
// CHECK-SAME:            cluster_id = 0 : i64,
// CHECK-SAME:            inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 0], cluster_id = 3 : i64>
// CHECK-NEXT:              ]

// CHECK:       [[OUT_CMX1:%.*]] = VPURT.DeclareBuffer <CMX_NN> [1] <10880> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x16x22x17xf16, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 10, 0], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 1], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 9], cluster_id = 1 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 0], cluster_id = 1 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 20, 1],
// CHECK-SAME:           offset = [0, 0, 0, 1],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 9],
// CHECK-SAME:           offset = [0, 0, 18, 1],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 9], offset = [0, 0, 0, 1], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 7],
// CHECK-SAME:           offset = [0, 0, 18, 10],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 7], offset = [0, 0, 0, 10], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 1],
// CHECK-SAME:           offset = [0, 0, 18, 1],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 16], cluster_id = 2 : i64>
// CHECK-NEXT:          ]

// CHECK:       [[OUT_CMX2:%.*]] = VPURT.DeclareBuffer <CMX_NN> [2] <10880> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x16x22x17xf16, #NHWC, [@CMX_NN, 2]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 2, 16], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 12, 16], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 16], cluster_id = 2 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 20, 1],
// CHECK-SAME:           offset = [0, 0, 2, 15],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 0], cluster_id = 3 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 16],
// CHECK-SAME:           offset = [0, 0, 2, 0],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 16], offset = [0, 0, 20, 0], cluster_id = 0 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 1],
// CHECK-SAME:           offset = [0, 0, 2, 15],
// CHECK-SAME:           cluster_id = 2 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 0], cluster_id = 1 : i64>
// CHECK-NEXT:          ]

// CHECK:       [[OUT_CMX3:%.*]] = VPURT.DeclareBuffer <CMX_NN> [3] <10880> -> !VPUIP.ITIBuffer<
// CHECK-NEXT:      1x16x22x17xf16, #NHWC, [@CMX_NN, 3]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 20, 1], offset = [0, 0, 2, 0], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 9], offset = [0, 0, 0, 1], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 7], offset = [0, 0, 0, 10], cluster_id = 3 : i64>,
// CHECK-NEXT:         #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 0, 0], cluster_id = 3 : i64>
// CHECK-NEXT:      ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 1],
// CHECK-SAME:           offset = [0, 0, 2, 1],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 2, 16], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 10, 1],
// CHECK-SAME:           offset = [0, 0, 12, 1],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 10, 1], offset = [0, 0, 12, 16], cluster_id = 2 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 8],
// CHECK-SAME:           offset = [0, 0, 2, 1],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 1], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 8],
// CHECK-SAME:           offset = [0, 0, 2, 9],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 8], offset = [0, 0, 20, 9], cluster_id = 1 : i64>
// CHECK-NEXT:          ]>
// CHECK-NEXT:          #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 1],
// CHECK-SAME:           offset = [0, 0, 2, 1],
// CHECK-SAME:           cluster_id = 3 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 1], offset = [0, 0, 20, 16], cluster_id = 0 : i64>
// CHECK-NEXT:          ]

// CHECK:        VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX1]], [[OUT_CMX2]], [[OUT_CMX3]]
// CHECK:     outputs([[OUT_CMX0]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX0]], [[OUT_CMX2]], [[OUT_CMX3]]
// CHECK:     outputs([[OUT_CMX1]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]], [[OUT_CMX3]]
// CHECK:     outputs([[OUT_CMX2]]

//CHECK:         VPUIP.NCEClusterTask {
// CHECK:     output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]], [[OUT_CMX2]]
// CHECK:     outputs([[OUT_CMX3]]

// -----

// CHECK-LABEL: @FourDWClusterTaskSOK
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType0 = !quant.uniform<u8:f16:0, {0.0036624862867243148:128,0.007217532980675791:128,0.042322833865296609:128,0.0037519759991589715:128,0.0073948338919994881:128,0.0089573402030795231:128,0.014193274928074257:128,0.010146608539656097:128,0.0052901366177727192:128,0.023849124534457339:128,0.0045484989297156237:128,0.0029279957799350515:128,0.0046218229275123745:128,0.0072132152669570028:128,0.02857208438948089:128,0.011937396666582892:128,0.025224551967546052:128,0.0060292753518796432:128,0.005454390890458051:128,0.0045290292478075214:128,0.0059620798802843284:128,0.0049756295540753529:128,0.0083414091783411362:128,0.0046041476960275683:128,0.0038931646767784568:128,0.014138247920017617:128,0.0084417445986878641:128,0.0063303610857795263:128,0.018603268791647518:128,0.062332589018578625:128,0.0072024198139415069:128,0.0068968408248003795:128,0.008046769628337785:128,0.0068004402459836471:128,0.049329922245998012:128,0.0067495257246728037:128,0.025976501726636701:128,0.0062383934563281489:128,0.0040290285559261545:128,0.0053795487272973157:128,0.014849628186693378:128,0.035341049643123852:128,0.0046583657171211995:128,0.0064138267554488841:128,0.0071043386178858137:128,0.0090361118316650384:128,0.052014283572926243:128,0.0047942470101749197:128,0.0049681429769478595:128,0.0072341425746094944:128,0.0218873659769694:128,0.0060195378228729843:128,0.016082307404162836:128,0.0075143276476392559:128,0.0051638009501438513:128,0.007972866413640041:128,0.0038064215697494208:128,0.0059307766895668182:128,0.020330337449616077:128,0.0032834243540670357:128,0.019986148909026502:128,0.020232869129554899:128,0.024704991134942747:128,0.0044083352182425701:128,0.024907423468197094:128,0.0031258799281774783:128,0.019521636588900698:128,0.009609512254303577:128,0.012374789574567009:128,0.0041377581802068972:128,0.0049908002217610679:128,0.0064016823675118245:128,0.0055642695987925808:128,0.043108336130777997:128,0.015441307834550446:128,0.0085692041060503789:128,0.005009850567462398:128,0.0066308806924258957:128,0.011756354219773236:128,0.011606741886512907:128,0.010951862615697524:128,0.0033850402224297619:128,0.0044465618975022261:128,0.0071446411749895881:128,0.011318085707870185:128,0.014506751883263681:128,0.0038469405735240261:128,0.035199432747036803:128,0.040727142259186389:128,0.0046945424640879915:128,0.0027322583338793586:128,0.0046014734342986465:128,0.0052412329935560041:128,0.0051573734657437195:128,0.0047896880729525696:128,0.011306174128663306:128,0.0054353461546056414:128,0.0051479292850868377:128,0.044744089537975838:128,0.044604026570039638:128,0.0045084446084265612:128,0.0067021535892112585:128,0.068990348367130055:128,0.010259167353312174:128,0.016896838767855776:128,0.01784057710684982:128,0.12893103057262945:128,0.0075389654028649427:128,0.0074017756125506233:128,0.046897170122931986:128,0.0068773473010343665:128,0.037301749809115541:128,0.026112250720753388:128,0.0063341498374938961:128,0.0053246970270194255:128,0.0086272749246335492:128,0.012482261190227434:128,0.0081099524217493387:128,0.0052486249044829724:128,0.0064642518174414538:128,0.0059139522851682173:128,0.0043014582465676701:128,0.0075986929968291641:128,0.022103037553675035:128,0.0067159332481085089:128,0.0097866979299807075:128,0.019208570555144664:128,0.0051108021362155096:128}>
!qElemType1 = !quant.uniform<u8:f16:1, {0.012679650736790078,0.0088509325887642654,0.001785417865304386,0.013014746647255094,0.0072207161024505015,0.0057379989063038542,0.0050717475367527383,0.0070249291027293485,0.011350748585719689,0.0019585400235419179,0.013752694223441329,0.015968934227438534,0.008925105076210171,0.0072272300720214845,0.0024838889346403235,0.0049831834493898877,0.0025285692775950711,0.011964638092938592,0.013224598940681009,0.010445049697277593,0.01285392536836512,0.0117491460313984,0.0079147264069201906,0.0096701537861543542,0.013943618886611041,0.004418591424530628,0.0099220098233690455,0.0095208598118202356,0.0048581263598273779,7.7644814463222731E-4,0.0087958588319666242,0.010552223990945256,0.0064019941816142959,0.0071454104255227486,0.0011597363387837129,0.0087246791989195587,0.0032697436856288535,0.0082136425317502482,0.012845136605057062,0.011476640140309054,0.0038081933470333323,0.0020070382193023081,0.0095260311575496893,0.0069194134543923771,0.006297496253368901,0.0066959829891429226,0.0010309690353917142,0.0099576510635076779,0.010925754846311083,0.01243127000098135,0.0026469001583024569,0.0086997209810743148,0.0058098316192626955,0.01085205452114928,0.01030415740667605,0.0075069053500306376,0.014588796391206629,0.008429828344606885,0.0046268729602589329,0.012260208877862668,0.0023927609125773113,0.0028241690467385684,0.0026601912928562537,0.013359493367812213,0.0043095710230808635,0.013482893214506261,0.0029009290769988413,0.0093032004786472687,0.0049605332168878294,0.013097328298232135,0.0093697117824180446,0.0095985019908231845,0.013058389401903339,0.0014802291112787583,0.0035873817462547153,0.0091267202414718334,0.014584920920577704,0.012644069335039924,0.0059420968972000417,0.0051122997321334544,0.0066340465171664366,0.012478245005888098,0.016404080858417585,0.0053457213383094936,0.0063293410282509004,0.0031673279463076123,0.013307671453438552,0.0017932187108432546,0.0020727900897755344,0.010207112630208333,0.011079003764133828,0.011203908920288086,0.011971709307502298,0.012528636408787147,0.011237553989186006,0.0052012321995753867,0.0094392355750588814,0.0078694268768908937,0.0018276142139060825,0.0018046914362439923,0.013108075833788105,0.0072822640923892751,0.0012064018670250387,0.008552738264495251,0.0047975077348596908,0.0048137206657260069,4.5556084198110244E-4,0.0070331087299421724,0.0065114166222366631,0.001282482053719315,0.011466080534691904,0.0016153688524283615,0.0023212077570896523,0.013336284487855201,0.012196069605210249,0.0087177800197227334,0.0049530674429500805,0.0070452788296867823,0.014796471128276751,0.012875244664210899,0.0087445090798770678,0.011763795216878255,0.0085144342160692402,0.0041161677416633154,0.0098459542966356471,0.0079605897267659501,0.0026574237673890359,0.011661134981641582}>

!Input0 = memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 0]>

!Input1 = memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 1]>

!Input2 = memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 2]>

!Input3 = memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 3]>

!OutputITI0 = !VPUIP.ITIBuffer<
        1x128x15x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 63488 : i64>}, [@CMX_NN, 0],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 14, 0], cluster_id = 0 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 13, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
            ]>
    ]>

!OutputITI1 = !VPUIP.ITIBuffer<
        1x128x16x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 63488 : i64>}, [@CMX_NN, 1],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
            #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 15, 0], cluster_id = 1 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 1, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 14, 0], cluster_id = 0 : i64>
            ]>,
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 14, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
            ]>
    ]>

!OutputITI2 = !VPUIP.ITIBuffer<
        1x128x16x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 63488 : i64>}, [@CMX_NN, 2],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 0, 0], cluster_id = 2 : i64>,
            #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 15, 0], cluster_id = 2 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 1, 0], cluster_id = 2 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 15, 0], cluster_id = 1 : i64>
            ]>,
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 14, 0], cluster_id = 2 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 0, 0], cluster_id = 3 : i64>
            ]>
    ]>

!OutputITI3 = !VPUIP.ITIBuffer<
        1x128x15x56x!qElemType1, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 63488 : i64>}, [@CMX_NN, 3],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 0, 0], cluster_id = 3 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 1, 0], cluster_id = 3 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 128, 1, 56], offset = [0, 0, 15, 0], cluster_id = 2 : i64>
            ]>
    ]>


module @FourDWClusterTaskSOK {
    net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x128x14x56xf16>
        DataInfo "input1" : tensor<1x128x14x56xf16>
        DataInfo "input2" : tensor<1x128x14x56xf16>
        DataInfo "input3" : tensor<1x128x14x56xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x128x15x56xf16>
        DataInfo "output1" : tensor<1x128x15x56xf16>
        DataInfo "output2" : tensor<1x128x15x56xf16>
        DataInfo "output3" : tensor<1x128x15x56xf16>
    }

func.func @main(%arg0:  memref<1x128x14x56xf16>, %arg1:  memref<1x128x14x56xf16>, %arg2:  memref<1x128x14x56xf16>, %arg3:  memref<1x128x14x56xf16>, %arg4:  memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 0]>, %arg5:  memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 1]>, %arg6:  memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 2]>, %arg7:  memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 3]>) -> (memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 0]>, memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 1]>, memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 2]>, memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 3]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1
    %input2 = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> !Input2
    %input3 = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> !Input3

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <10880> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <10880> ->  !OutputITI1
    %output2 = VPURT.DeclareBuffer <CMX_NN> [2] <10880> ->  !OutputITI2
    %output3 = VPURT.DeclareBuffer <CMX_NN> [3] <10880> ->  !OutputITI3

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 1]>
    %weights2 = VPURT.DeclareBuffer <CMX_NN> [2] <2816> -> memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 2]>
    %weights3 = VPURT.DeclareBuffer <CMX_NN> [3] <2816> -> memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 3]>

    VPURT.Task {
        VPUIP.NCEClusterTask {
                is_small_kernel_optimized,
                kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<DWCONV>
            }
            input(%input0: !Input0)
            weights(%weights0: memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 0]>)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1, %output2, %output3 : !OutputITI1, !OutputITI2, !OutputITI3)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over K
                DPUTask {cluster_id = 0 : i64,inEnd = [55, 14, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, workload_id = 0 : i64}
                DPUTask {cluster_id = 0 : i64, inEnd = [55, 14, 63], inStart = [0, 0, 32], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, workload_id = 1 : i64}
                DPUTask {cluster_id = 0 : i64, inEnd = [55, 14, 95], inStart = [0, 0, 64], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 95], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, workload_id = 2 : i64}
                DPUTask {cluster_id = 0 : i64, inEnd = [55, 14, 127], inStart = [0, 0, 96], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 127], outStart = [0, 0, 96], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>, workload_id = 3 : i64}
            } PPE : {
                PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                is_small_kernel_optimized,
                kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<DWCONV>
            }
            input(%input1: !Input1)
            weights(%weights1: memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 1]>)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0, %output2, %output3: !OutputITI0, !OutputITI2, !OutputITI3)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over K
                DPUTask {cluster_id = 1 : i64, inEnd = [55, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 0 : i64}
                DPUTask {cluster_id = 1 : i64, inEnd = [55, 15, 63], inStart = [0, 0, 32], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 1 : i64}
                DPUTask {cluster_id = 1 : i64, inEnd = [55, 15, 95], inStart = [0, 0, 64], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 95], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 2 : i64}
                DPUTask {cluster_id = 1 : i64, inEnd = [55, 15, 127], inStart = [0, 0, 96], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 127], outStart = [0, 0, 96], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 3 : i64}
            } PPE : {
                PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                is_small_kernel_optimized,
                kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<DWCONV>
            }
            input(%input2: !Input2)
            weights(%weights2: memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 2]>)
            parent_input(%input2: !Input2)
            parent_output(%output2: !OutputITI2)
            output_ITI_buff(%output0, %output1, %output3: !OutputITI0, !OutputITI1, !OutputITI3)
            outputs(%output2: !OutputITI2)
            -> !OutputITI2
            variants : { // Workloads split over K
                DPUTask {cluster_id = 2 : i64, inEnd = [55, 15, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 0 : i64}
                DPUTask {cluster_id = 2 : i64, inEnd = [55, 15, 63], inStart = [0, 0, 32], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 1 : i64}
                DPUTask {cluster_id = 2 : i64, inEnd = [55, 15, 95], inStart = [0, 0, 64], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 95], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 2 : i64}
                DPUTask {cluster_id = 2 : i64, inEnd = [55, 15, 127], inStart = [0, 0, 96], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 127], outStart = [0, 0, 96], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>, workload_id = 3 : i64}
            } PPE : {
                PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
            is_small_kernel_optimized,
                kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
                kernel_size = [3, 3],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<DWCONV>
            }
            input(%input3: !Input3)
            weights(%weights3: memref<128x16x1x1x!qElemType0, #NHWC, [@CMX_NN, 3]>)
            parent_input(%input3: !Input3)
            parent_output(%output3: !OutputITI3)
            output_ITI_buff(%output0, %output1, %output2: !OutputITI0, !OutputITI1, !OutputITI2)
            outputs(%output3: !OutputITI3)
            -> !OutputITI3
            variants : { // Workloads split over K-channel
                DPUTask {cluster_id = 3 : i64, inEnd = [55, 14, 31], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 31], outStart = [0, 0, 0], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, workload_id = 0 : i64}
                DPUTask {cluster_id = 3 : i64, inEnd = [55, 14, 63], inStart = [0, 0, 32], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 63], outStart = [0, 0, 32], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, workload_id = 1 : i64}
                DPUTask {cluster_id = 3 : i64, inEnd = [55, 14, 95], inStart = [0, 0, 64], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 95], outStart = [0, 0, 64], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, workload_id = 2 : i64}
                DPUTask {cluster_id = 3 : i64, inEnd = [55, 14, 127], inStart = [0, 0, 96], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [55, 13, 127], outStart = [0, 0, 96], pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, workload_id = 3 : i64}
            } PPE : {
                PPETask {ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>}
            }
    }
    return %arg4, %arg5, %arg6, %arg7: memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 0]>, memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 1]>, memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 2]>, memref<1x128x15x56x!qElemType1, #NHWC, [@CMX_NN, 3]>
}

//CHECK:    [[INPUT_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 0]>
//CHECK:    [[INPUT_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 1]>
//CHECK:    [[INPUT_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <0> -> memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 2]>
//CHECK:    [[INPUT_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 3]>

//CHECK:    [[OUT_CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <10880>
//CHECK:    [[OUT_CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <10880>
//CHECK:    [[OUT_CMX2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <10880>
//CHECK:    [[OUT_CMX3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <10880>

//CHECK:    [[WTS_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2816> -> memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 0]>
//CHECK:    [[WTS_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <2816> -> memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 1]>
//CHECK:    [[WTS_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [2] <2816> -> memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 2]>
//CHECK:    [[WTS_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <2816> -> memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 3]>


//CHECK:      [[NCE_TASK:%.+]] = VPUIP.NCEClusterTask {is_small_kernel_optimized,
//CHECK-SAME:      kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
//CHECK-SAME:      kernel_size = [3, 3],
//CHECK-SAME:      kernel_strides = [1, 1],
//CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>}
//CHECK:      input([[INPUT_0]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK:      weights([[WTS_0]] : memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 0]>)
//CHECK:      parent_input([[INPUT_0]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 0]>)
//CHECK:      parent_output([[OUT_CMX0]]
//CHECK:      output_ITI_buff([[OUT_CMX1]], [[OUT_CMX2]], [[OUT_CMX3]]
//CHECK:      outputs([[OUT_CMX0]]

//CHECK:     VPUIP.NCEClusterTask {is_small_kernel_optimized,
//CHECK-SAME:      kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
//CHECK-SAME:      kernel_size = [3, 3],
//CHECK-SAME:      kernel_strides = [1, 1],
//CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>}
//CHECK:      input([[INPUT_1]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 1]>)
//CHECK:      weights([[WTS_1]] : memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 1]>)
//CHECK:      parent_input([[INPUT_1]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 1]>)
//CHECK:      parent_output([[OUT_CMX1]]
//CHECK:      output_ITI_buff([[OUT_CMX0]], [[OUT_CMX2]], [[OUT_CMX3]]
//CHECK:      outputs([[OUT_CMX1]]

//CHECK:    VPUIP.NCEClusterTask {is_small_kernel_optimized,
//CHECK-SAME:      kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>,
//CHECK-SAME:      kernel_size = [3, 3],
//CHECK-SAME:      kernel_strides = [1, 1],
//CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>}
//CHECK:      input([[INPUT_2]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 2]>)
//CHECK:      weights([[WTS_2]] : memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 2]>)
//CHECK:      parent_input([[INPUT_2]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 2]>)
//CHECK:      parent_output([[OUT_CMX2]]
//CHECK:      output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]], [[OUT_CMX3]]
//CHECK:      outputs([[OUT_CMX2]]

//CHECK:   VPUIP.NCEClusterTask {is_small_kernel_optimized,
//CHECK-SAME:      kernel_padding = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
//CHECK-SAME:      kernel_size = [3, 3],
//CHECK-SAME:      kernel_strides = [1, 1],
//CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>}
//CHECK:      input([[INPUT_3]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 3]>)
//CHECK:      weights([[WTS_3]] : memref<128x16x1x1x!qElemType1, #NHWC, [@CMX_NN, 3]>)
//CHECK:      parent_input([[INPUT_3]] : memref<1x128x14x56xf16, #NHWC, [@CMX_NN, 3]>)
//CHECK:      parent_output([[OUT_CMX3]]
//CHECK:      output_ITI_buff([[OUT_CMX0]], [[OUT_CMX1]], [[OUT_CMX2]]
//CHECK:      outputs([[OUT_CMX3]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x2x121xf16, #NHWC, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x2x121xf16, #NHWC, [@CMX_NN, 1]
>

!OutputITI0 = !VPUIP.ITIBuffer<
    1x16x4x121xf16, #NHWC, [@CMX_NN, 0],
    inwardHaloRegions = [
          #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 2, 0], cluster_id = 0 : i64>
    ],
    outwardHaloRegions = [
          #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
         ]>
    ]>


!OutputITI1 = !VPUIP.ITIBuffer<
    1x16x4x121xf16, #NHWC, [@CMX_NN, 1],
    inwardHaloRegions = [
          #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    ],
    outwardHaloRegions = [
          #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 2, 0], cluster_id = 1 : i64, inwardHaloRegions = [
              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 2, 0], cluster_id = 0 : i64>
          ]>
    ]>

// CHECK-LABEL: @TwoEltWiseNCEClusterTasksSOH
module @TwoEltWiseNCEClusterTasksSOH {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x16x2x121xf16>
        DataInfo "input1" : tensor<1x16x2x121xf16>
    }
    outputsInfo : {
        DataInfo "output0" : tensor<1x16x4x121xf16>
        DataInfo "output1" : tensor<1x16x4x121xf16>
    }

func.func @main(%arg0:  memref<1x16x2x121xf16, #NHWC, [@CMX_NN, 0]>, %arg1:  memref<1x16x2x121xf16, #NHWC, [@CMX_NN, 1]>,
                %arg2:  memref<1x16x4x121xf16, #NHWC, [@CMX_NN, 0]>, %arg3: memref<1x16x4x121xf16, #NHWC, [@CMX_NN, 1]>)
-> (memref<1x16x4x121xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x4x121xf16, #NHWC, [@CMX_NN, 1]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <15488> -> !Input0
    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <15488> -> !Input0
    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !OutputITI0

    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <15488> -> !Input1
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <15488> -> !Input1
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !OutputITI1

    VPURT.Task {
        VPUIP.NCEClusterTask {
                eltwise_type = #VPU.eltwise_type<ADD>,
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }
            input(%input0: !Input0)
            weights(%weights0 : !Input0)
            parent_input(%input0: !Input0)
            parent_output(%output0: !OutputITI0)
            output_ITI_buff(%output1 : !OutputITI1)
            outputs(%output0: !OutputITI0)
            -> !OutputITI0
            variants : { // Workloads split over H
                DPUTask {
                    cluster_id = 0 : i64,
                    inEnd = [120, 1, 15], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                    outEnd = [120, 1, 15], outStart = [0, 0, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            } PPE : {
            }
    }

    VPURT.Task {
        VPUIP.NCEClusterTask {
                eltwise_type = #VPU.eltwise_type<ADD>,
                task_type = #VPUIP.nce_task_type<ELTWISE>
            }
            input(%input1: !Input1)
            weights(%weights1 : !Input1)
            parent_input(%input1: !Input1)
            parent_output(%output1: !OutputITI1)
            output_ITI_buff(%output0 : !OutputITI0)
            outputs(%output1: !OutputITI1)
            -> !OutputITI1
            variants : { // Workloads split over H
                DPUTask {
                    cluster_id = 1 : i64,
                    inEnd = [120, 1, 15], inStart = [0, 0, 0],
                    mpe_mode = #VPU.mpe_mode<CUBOID_8x16>,
                    outEnd = [120, 3, 15], outStart = [0, 2, 0],
                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
                }
            } PPE : {
            }
    }

    return %arg2, %arg3: memref<1x16x4x121xf16, #NHWC, [@CMX_NN, 0]>, memref<1x16x4x121xf16, #NHWC, [@CMX_NN, 1]>

// CHECK:       [[OUT_CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !VPUIP.ITIBuffer<
// CHECK:           1x16x4x121xf16, #NHWC, [@CMX_NN, 0],
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK-NEXT:          #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 2, 0], cluster_id = 0 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:               #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:              shape = [1, 16, 2, 121],
// CHECK-SAME:              offset = [0, 0, 0, 0],
// CHECK-SAME:              cluster_id = 0 : i64,
// CHECK-SAME:              inwardHaloRegions = [
// CHECK-NEXT:                  #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK-NEXT:              ]>
// CHECK:           ]>

// CHECK:       [[OUT_CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !VPUIP.ITIBuffer<
// CHECK:           1x16x4x121xf16, #NHWC, [@CMX_NN, 1]
// CHECK-NEXT:      inwardHaloRegions = [
// CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
// CHECK:           ],
// CHECK-NEXT:      outwardHaloRegions = [
// CHECK:              #VPUIP.OutwardHaloRegionAttr<
// CHECK-SAME:           shape = [1, 16, 2, 121],
// CHECK-SAME:           offset = [0, 0, 2, 0],
// CHECK-SAME:           cluster_id = 1 : i64,
// CHECK-SAME:           inwardHaloRegions = [
// CHECK-NEXT:              #VPUIP.HaloRegionAttr<shape = [1, 16, 2, 121], offset = [0, 0, 2, 0], cluster_id = 0 : i64>
// CHECK-NEXT:          ]>

// CHECK:        VPUIP.NCEClusterTask {
// CHECK-SAME:         eltwise_type = #VPU.eltwise_type<ADD>, task_type = #VPUIP.nce_task_type<ELTWISE>
// CHECK:          output_ITI_buff([[OUT_CMX1]]
// CHECK:          outputs([[OUT_CMX0]]

// CHECK:         VPUIP.NCEClusterTask {
// CHECK-SAME:         eltwise_type = #VPU.eltwise_type<ADD>, task_type = #VPUIP.nce_task_type<ELTWISE>
// CHECK:           output_ITI_buff([[OUT_CMX0]]
// CHECK:           outputs([[OUT_CMX1]]

}

}
