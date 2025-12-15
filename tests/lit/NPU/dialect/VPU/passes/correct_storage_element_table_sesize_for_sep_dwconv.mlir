//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --correct-storage-element-table-sesize-for-sep-dwconv %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @CorrectSESize {
    config.Resources 4 of @NCE at 6.000000e+02 MHz

// CHECK-LABEL: @DWConvWithSEPSOK
func.func @DWConvWithSEPSOK(%arg0: tensor<1x160x1x1xf16, {order = #NHWC}>) -> tensor<1x160x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<160x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x160x2x2xi1> = dense<1> : tensor<1x160x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16, seDepth = 10, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 160, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 160, 2, 2]>
    } -> tensor<1x10x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 160, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x160x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x160x2x2xi1>,
                           storage_element_table=tensor<1x10x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 160, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [160, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x160x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x160x2x2xf16, {order = #NHWC}>

    // CHECK:       VPU.StorageElementTable
    // CHECK-SAME:     seDepth = 4 : i64, seSize = [64, 32, 32, 32]
    // CHECK-SAME:  -> tensor<1x4x2x2xi32, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @CorrectSESize {
    config.Resources 4 of @NCE at 6.000000e+02 MHz

// CHECK-LABEL: @DWConvWithSEPSOH
func.func @DWConvWithSEPSOH(%arg0: tensor<1x64x4x4xf16, {order = #NHWC}>) -> tensor<1x64x8x8xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x64x8x8xi1> = dense<1> : tensor<1x64x8x8xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16, seDepth = 4, seSize = [16, 16, 16, 16], dataShape = [1, 64, 4, 4],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 8, 8]>
    } -> tensor<1x4x8x8xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 8, 8]>
    } -> !VPU.SparseTensor<data=tensor<1x64x4x4xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x8x8xi1>,
                           storage_element_table=tensor<1x4x8x8xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 8, 8]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x64x8x8xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x8x8xf16, {order = #NHWC}>

    // CHECK:       VPU.StorageElementTable
    // CHECK-SAME:     seDepth = 1 : i64, seSize = [64]
    // CHECK-SAME:  -> tensor<1x1x8x8xi32, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DWConvWithSEPNoMCStrategy
func.func @DWConvWithSEPNoMCStrategy(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x64x2x2xi1> = dense<1> : tensor<1x64x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16, seDepth = 4, seSize = [16, 16, 16, 16], dataShape = [1, 64, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]>
    } -> tensor<1x4x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x64x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x2x2xi1>,
                           storage_element_table=tensor<1x4x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x64x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x2x2xf16, {order = #NHWC}>

    // CHECK:       VPU.StorageElementTable
    // CHECK-SAME:     seDepth = 1 : i64, seSize = [64]
    // CHECK-SAME:  -> tensor<1x1x2x2xi32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Consider the following scenario, with tiling over input channel and depth = 1:
// %input_data (128) -> Slice on IC -> (64) ---> GroupSparseOp
// %smap (64ch) ---------------------------------/  |
// %se_table (se_sz = 128, depth = 1) --------------/
//    \----------------------------- (... other tile on channles of Dw.Conv op ...)
//
// SETable should be, basically, "DUPLICATED" for each channel tile in this case.
// The logic that moves Slice before GroupSparseOp will put a Slice on the SETable
// branch, but it will be folded, leaving the SETable with larger seSize.

// CHECK-LABEL: @DWConvWithSEPChannelSliceWithDepth1
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @DWConvWithSEPChannelSliceWithDepth1(%arg0: tensor<1x64x1x1xf16, {order = #NHWC}>) -> tensor<1x64x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x64x2x2xi1> = dense<1> : tensor<1x64x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [128], dataShape = [1, 128, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x64x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x64x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [64, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x64x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x2x2xf16, {order = #NHWC}>

    // CHECK:       [[SMAP:%.+]] = const.Declare tensor<1x64x2x2xi1> = dense<true> : tensor<1x64x2x2xi1>
    // CHECK:       [[SE_TABLE:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 1, 1]
    // CHECK-SAME:         seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                  scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]>
    // CHECK-SAME:         seDepth = 1 : i64, seSize = [64]
    // CHECK-SAME:      -> tensor<1x1x2x2xi32, {order = #NHWC}>

    // CHECK:       [[INPUT:%.+]] = VPU.GroupSparseTensor([[ARG0]], [[SMAP]], [[SE_TABLE]])
    // CHECK-SAME:    -> !VPU.SparseTensor<data=tensor<1x64x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:         sparsity_map=tensor<1x64x2x2xi1>,
    // CHECK-SAME:         storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
    // CHECK-SAME:         #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                            scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 64, 2, 2]>>

    // CHECK:       VPU.NCE.DepthConvolution([[INPUT]]
}
