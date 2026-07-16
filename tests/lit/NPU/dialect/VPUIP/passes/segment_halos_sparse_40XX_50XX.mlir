//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --segment-halos %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!Input0 = memref<
    1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 0]
>

!Input1 = memref<
    1x16x16x32xf16, {order = #NHWC}, [@CMX_NN, 1]
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

func.func @main(%arg0:  memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg1:  memref<1x16x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>, %arg2:  memref<1x32x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, %arg3:  memref<1x32x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) -> (memref<1x32x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>) {
    %input0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> !Input0
    %input1 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> !Input1

    %output0 = VPURT.DeclareBuffer <CMX_NN> [0] <17408> ->  !OutputITI0
    %output1 = VPURT.DeclareBuffer <CMX_NN> [1] <17408> ->  !OutputITI1

    %weights0 = VPURT.DeclareBuffer <CMX_NN> [0] <34816> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %weights1 = VPURT.DeclareBuffer <CMX_NN> [1] <34816> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>

    %output_sm0 = VPURT.DeclareBuffer <CMX_NN> [0] <39680> -> !OutputITISparse0
    %output_sm1 = VPURT.DeclareBuffer <CMX_NN> [1] <39680> -> !OutputITISparse1

    VPURT.Task {
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%input0: !Input0)
            weights(%weights0: memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
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
        VPUIP.NCEClusterTask {resultSegmentSizes = array<i32: 1, 1, 0, 0, 0, 0>} <{
                kernel_padding = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                kernel_size = [1, 1],
                kernel_strides = [1, 1],
                task_type = #VPUIP.nce_task_type<CONV>
            }>
            input(%input1: !Input1)
            weights(%weights1: memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 1]>)
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

    return %arg2, %arg3: memref<1x32x17x32xf16, {order = #NHWC}, [@CMX_NN, 0]>, memref<1x32x17x32xf16, {order = #NHWC}, [@CMX_NN, 1]>
}

}

// CHECK:       [[OUT_CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <17408> -> !VPUIP.ITIBuffer<
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

// CHECK:       [[OUT_CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <17408> -> !VPUIP.ITIBuffer<
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

// CHECK:       [[OUT_SPARSE_CMX0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <39680> -> !VPUIP.ITIBuffer<
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

// CHECK:       [[OUT_SPARSE_CMX1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <39680> -> !VPUIP.ITIBuffer<
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


// CHECK:        VPUIP.NCEClusterTask <{
// CHECK:          parent_output_sparsity_map([[OUT_SPARSE_CMX0]]
// CHECK:          output_ITI_buff([[OUT_CMX1]]
// CHECK:          outputs([[OUT_CMX0]]
// CHECK:          output_sparsity_map([[OUT_SPARSE_CMX0]]

// CHECK:         VPUIP.NCEClusterTask <{
// CHECK:           parent_output_sparsity_map([[OUT_SPARSE_CMX1]]
// CHECK:           output_ITI_buff([[OUT_CMX0]]
// CHECK:           outputs([[OUT_CMX1]]
// CHECK:           output_sparsity_map([[OUT_SPARSE_CMX1]]
