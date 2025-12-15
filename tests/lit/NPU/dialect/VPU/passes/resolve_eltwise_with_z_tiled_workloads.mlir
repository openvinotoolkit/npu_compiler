//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --resolve-eltwise-with-z-tiled-workloads %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoChangeEltwise
func.func @NoChangeEltwise(%arg0: tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>, %arg1: tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>
        -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>
        -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
        VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 96, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
    }

    %3 = VPU.Copy(%2) {out_mem_space = @DDR} : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    return %3 : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[INPUT1:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[INPUT2:%.+]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 96, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 96, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:       }

    // CHECK:       [[OUTPUT:%.+]] = VPU.Copy([[ELTWISE]]) {out_mem_space = @DDR} : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       return [[OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseWorkloadsTiledOverZ
func.func @EltwiseWorkloadsTiledOverZ(%arg0: tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>, %arg1: tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>
        -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>
        -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
        VPU.DPU.Workload outOffsets [0, 32, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
        VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 32, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
    }

    %3 = VPU.Copy(%2) {out_mem_space = @DDR} : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    return %3 : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[INPUT1:%.+]] = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[INPUT2:%.+]] = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}> -> tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 16, 16] : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT2:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 32, 16, 16] : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[SLICE1_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE1_INPUT1]], [[SLICE1_INPUT2]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:       }
    // CHECK:       [[SLICE1_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE1_ELTWISE]]) {out_mem_space = @DDR} : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE2_INPUT1:%.+]] = VPU.Slice [[INPUT1]] [0, 32, 0, 0] [1, 32, 16, 16] : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT2:%.+]] = VPU.Slice [[INPUT2]] [0, 32, 0, 0] [1, 32, 16, 16] : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[SLICE2_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE2_INPUT1]], [[SLICE2_INPUT2]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:       }
    // CHECK:       [[SLICE2_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE2_ELTWISE]]) {out_mem_space = @DDR} : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE3_INPUT1:%.+]] = VPU.Slice [[INPUT1]] [0, 64, 0, 0] [1, 32, 16, 16] : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[SLICE3_INPUT2:%.+]] = VPU.Slice [[INPUT2]] [0, 64, 0, 0] [1, 32, 16, 16] : tensor<1x96x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> to tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[SLICE3_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE3_INPUT1]], [[SLICE3_INPUT2]])
    // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    // CHECK:       }
    // CHECK:       [[SLICE3_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE3_ELTWISE]]) {out_mem_space = @DDR} : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[SLICE1_ELTWISE_COPY]], [[SLICE2_ELTWISE_COPY]], [[SLICE3_ELTWISE_COPY]])
    // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0], [0, 64, 0, 0]]}
    // CHECK-SAME:            : tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:              tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:              tensor<1x32x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseWorkloadsTiledOverZClustering
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
// CHECK-SAME:    [[ARG1:%.+]]: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>)

func.func @EltwiseWorkloadsTiledOverZClustering(%arg0: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>, %arg1: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    %3 = VPU.Copy(%2) {out_mem_space = @DDR} : !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}>
    -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    return %3 : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[INPUT1:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    // CHECK:       [[INPUT2:%.+]] = VPU.Copy([[ARG1]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[INPUT1_DDR:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:          -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[INPUT2_DDR:%.+]] = VPU.Copy([[INPUT2]])
    // CHECK-SAME:          -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT1]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT2]])
    // CHECK-SAME:               -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE1_INPUT1_COPY]], [[SLICE1_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }

    // CHECK:       [[SLICE1_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE1_ELTWISE]])
    // CHECK-SAME:          -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE2_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT1]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT2]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE2_INPUT1_COPY]], [[SLICE2_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64, alignment = [1, 16, 1, 1]}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE2_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE2_ELTWISE]])
    // CHECK-SAME:          -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[SLICE1_ELTWISE_COPY]], [[SLICE2_ELTWISE_COPY]])
    // CHECK{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]}
    // CHECK-SAME:    : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:      tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseWorkloadsTiledOverZSOH
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
// CHECK-SAME:    [[ARG1:%.+]]: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>)
func.func @EltwiseWorkloadsTiledOverZSOH(%arg0: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>, %arg1: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 8, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    %3 = VPU.Copy(%2) {out_mem_space = @DDR} : !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
            -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    return %3 : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[INPUT1:%.+]] = VPU.Copy([[ARG0]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT2:%.+]] = VPU.Copy([[ARG1]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[INPUT1_DDR:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:          -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[INPUT2_DDR:%.+]] = VPU.Copy([[INPUT2]])
    // CHECK-SAME:          -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT1]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT2]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE1_INPUT1_COPY]], [[SLICE1_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE1_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE1_ELTWISE]])
    // CHECK-SAME:          -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE2_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT1]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT2]])
    // CHECK-SAME:          -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE2_INPUT1_COPY]], [[SLICE2_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE2_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE2_ELTWISE]])
    // CHECK-SAME:          -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[SLICE1_ELTWISE_COPY]], [[SLICE2_ELTWISE_COPY]])
    // CHECK{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]}
    // CHECK-SAME:    : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:    tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseWorkloadsTiledOverZHKSwitch
func.func @EltwiseWorkloadsTiledOverZHKSwitch(%arg0: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>, %arg1: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 8, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    %3 = VPU.Copy(%2) {out_mem_space = @DDR}
         : !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
         -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    return %3 : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[INPUT1:%.+]] = VPU.Copy(%arg0)
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT2:%.+]] = VPU.Copy(%arg1)
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[INPUT1_DDR:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:                  -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[INPUT2_DDR:%.+]] = VPU.Copy([[INPUT2]])
    // CHECK-SAME:                  -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT1]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT2]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE1_INPUT1_COPY]], [[SLICE1_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE1_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE1_ELTWISE]])
    // CHECK-SAME:                  -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE2_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT1]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT2]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE2_INPUT1_COPY]], [[SLICE2_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED|MULTICASTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE2_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE2_ELTWISE]])
    // CHECK-SAME:                   -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[SLICE1_ELTWISE_COPY]], [[SLICE2_ELTWISE_COPY]])
    // CHECK{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]}
    // CHECK-SAME:    : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:    tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseWorkloadsTiledOverZSOHWithNCEConsumer
func.func @EltwiseWorkloadsTiledOverZSOHWithNCEConsumer(%arg0: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>, %arg1: tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>) -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> {
    %0 = VPU.Copy(%arg0) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
                -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %1 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
            -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 0, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
            VPU.DPU.Workload outOffsets [0, 64, 8, 0] outSizes [1, 64, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    %3 = VPU.NCE.Eltwise(%2, %2) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEStub<>
                } -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
            VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 128, 8, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 1 : i64}
        }

    %4 = VPU.Copy(%3) {out_mem_space = @DDR} : !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    return %4 : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[INPUT1:%.+]] = VPU.Copy(%arg0)
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[INPUT2:%.+]] = VPU.Copy(%arg1)
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[INPUT1_DDR:%.+]] = VPU.Copy([[INPUT1]])
    // CHECK-SAME:                  -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[INPUT2_DDR:%.+]] = VPU.Copy([[INPUT2]])
    // CHECK-SAME:                  -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE1_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT1]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    // CHECK:       [[SLICE1_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 0, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE1_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE1_INPUT2]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE1_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE1_INPUT1_COPY]], [[SLICE1_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE1_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE1_ELTWISE]])
    // CHECK-SAME:                  -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[SLICE2_INPUT1:%.+]] = VPU.Slice [[INPUT1_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT1_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT1]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_INPUT2:%.+]] = VPU.Slice [[INPUT2_DDR]] [0, 64, 0, 0] [1, 64, 16, 16] : tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}> to tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK:       [[SLICE2_INPUT2_COPY:%.+]] = VPU.Copy([[SLICE2_INPUT2]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[SLICE2_ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE2_INPUT1_COPY]], [[SLICE2_INPUT2_COPY]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x64x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 64, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }
    // CHECK:       [[SLICE2_ELTWISE_COPY:%.+]] = VPU.Copy([[SLICE2_ELTWISE]])
    // CHECK-SAME:                  -> tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[SLICE1_ELTWISE_COPY]], [[SLICE2_ELTWISE_COPY]])
    // CHECK{LITERAL}:   {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]}
    // CHECK-SAME:    : tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>,
    // CHECK-SAME:      tensor<1x64x16x16xf16, {mem_space = @DDR, order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       [[CONCAT_CMX:%.+]] = VPU.Copy([[CONCAT]])
    // CHECK-SAME:                  -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[CONCAT_CMX]], [[CONCAT_CMX]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
    // CHECK-SAME:        -> !VPU.DistributedTensor<1x128x16x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}> {
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 128, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    // CHECK:           VPU.DPU.Workload outOffsets [0, 0, 8, 0] outSizes [1, 128, 8, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16> attributes {cluster_id = 1 : i64}
    // CHECK:       }

    // CHECK:       [[OUTPUT_DDR:%.+]] = VPU.Copy([[OUTPUT]])
    // CHECK-SAME:                  -> tensor<1x128x16x16xf16, {mem_space = @DDR, order = #NHWC}>

    // CHECK:       return [[OUTPUT_DDR]]
}
