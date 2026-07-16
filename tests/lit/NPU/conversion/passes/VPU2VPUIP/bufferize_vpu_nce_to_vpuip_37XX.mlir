//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEConvolution
func.func @SuperdenseNCEConvolution(%arg0: tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                     %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                     %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
                     ) -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
    %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,

        strides = [1, 1]
    } : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}> -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] pad [0, 0, 0, 13] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       VPUIP.NCEClusterTask <{
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  }>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEMaxPool
func.func @SuperdenseNCEMaxPool(%arg0: tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                 %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
                 ) -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
    %0 = VPU.NCE.MaxPool(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] pad [0, 0, 0, 13] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       VPUIP.NCEClusterTask <{
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:  }>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEAveragePool
func.func @SuperdenseNCEAveragePool(%arg0: tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        minimumHardwareExecutionCost = 708 : i64,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] pad [0, 0, 0, 13] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 708 : i64} <{
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<AVEPOOL>
    // CHECK-SAME:  }>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SuperdenseNCEEltwise
func.func @SuperdenseNCEEltwise(%arg0: tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                 %arg1: tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>
                 ) -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        minimumHardwareExecutionCost = 585 : i64,
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] pad [0, 0, 0, 13] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 585 : i64} <{
    // CHECK-SAME:      eltwise_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  }>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateNearest(
    %arg0: tensor<1x64x5x10xf16, {order = #NHWC}>,           // data
    %arg1: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,  // weights
    %arg2: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>          // weight table
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %sparsityMap = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 1,
        seSize = [64],
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            nearest_mode = <FLOOR>,
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x10x20xi1>,
            storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %input_cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
        data=tensor<1x64x5x10xf16, {order = #NHWC}>,
        sparsity_map=tensor<1x64x10x20xi1>,
        storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>>
        -> !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            sparsity_map=tensor<1x64x10x20xi1, {mem_space = @CMX_NN}>,
            storage_element_table=tensor<1x1x10x20xi32, {mem_space = @CMX_NN, order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %task = VPU.NCE.Interpolate(%input_cmx, %arg1, %arg2) rawFilterShape [64, 64, 1, 1] {

        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>
    {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %task : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:   VPUIP.GroupSparseBuffer
    // CHECK:   {seAttr = #VPU.SEInterpolate<mode = <NEAREST>,
    // CHECK-SAME:    coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:    nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 10, 20]>}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @InterpolateBilinear(
    %arg0: tensor<1x64x5x10xf16, {order = #NHWC}>,           // data
    %arg1: tensor<64x64x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,  // weights
    %arg2: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>          // weight table
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {
    %sparsityMap = const.Declare tensor<1x64x11x21xi1> = dense<1> : tensor<1x64x11x21xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 1,
        seSize = [64],
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x11x21xi1>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %input_cmx = VPU.Copy(%input) {out_mem_space = @CMX_NN} : !VPU.SparseTensor<
        data=tensor<1x64x5x10xf16, {order = #NHWC}>,
        sparsity_map=tensor<1x64x11x21xi1>,
        storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
        #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>>
        -> !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC, mem_space = @CMX_NN}>,
            sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC, mem_space = @CMX_NN}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %task = VPU.NCE.Interpolate(%input_cmx, %arg1, %arg2) rawFilterShape [64, 64, 2, 2] {

        ppe = #VPU.PPEStub<>,
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2]
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>
    {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %task : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:   VPUIP.GroupSparseBuffer
    // CHECK:   {seAttr = #VPU.SEInterpolate<mode = <BILINEAR>,
    // CHECK-SAME:    coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:    scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:    offsets = [0, 0, 0, 0], sizes = [1, 64, 11, 21]>}
}
