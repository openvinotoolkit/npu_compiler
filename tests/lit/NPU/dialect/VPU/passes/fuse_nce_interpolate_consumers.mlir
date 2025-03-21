//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --fuse-nce-interpolate-consumers %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @FuseInterpolateNearestWithConv([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x32x6x6xf16, {order = #NHWC}> {
func.func @FuseInterpolateNearestWithConv(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x32x6x6xf16, {order = #NHWC}> {
    %interp_weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %interp_weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %interp_sparsity_map = const.Declare tensor<1x16x6x6xi1> = dense<1> : tensor<1x16x6x6xi1>

    %interp_storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 3, 3],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 6, 6]>
    } -> tensor<1x1x6x6xi32, {order = #NHWC}>

    %interp_input = VPU.GroupSparseTensor(%arg0, %interp_sparsity_map, %interp_storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 6, 6]>
    } -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x6x6xi1>,
                           storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 6, 6]>>

    %interp_output = VPU.NCE.Interpolate(%interp_input, %interp_weights, %interp_weights_table) {
        strides = [1, 1],
        rawFilterShape = [16, 16, 1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x6x6xf16, {order = #NHWC}>

    %conv_weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %conv_output = VPU.NCE.Convolution(%interp_output, %conv_weights, %conv_weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x32x6x6xf16, {order = #NHWC}>

    return %conv_output : tensor<1x32x6x6xf16, {order = #NHWC}>


    // CHECK-DAG:  [[CST_WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:  [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:  [[INPUT_SPARSITY_MAP:%.+]] = const.Declare tensor<1x16x6x6xi1> = dense<true> : tensor<1x16x6x6xi1>

    // CHECK:      [[INPUT_SE_TABLE:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:     dataElemType = i32,
    // CHECK-SAME:     dataShape = [1, 16, 3, 3],
    // CHECK-SAME:     seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                 scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                 nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 6, 6]>,
    // CHECK-SAME:     seDepth = 1 : i64,
    // CHECK-SAME:     seSize = 16 : i64
    // CHECK-SAME:   } -> tensor<1x1x6x6xi32, {order = #NHWC}>
    // CHECK:      [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SPARSITY_MAP]], [[INPUT_SE_TABLE]]) {
    // CHECK-SAME:     seAttr = #VPU.SEInterpolate<mode = <NEAREST>,
    // CHECK-SAME:     coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:     scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:     nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 6, 6]>
    // CHECK-SAME:   } -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                        sparsity_map=tensor<1x16x6x6xi1>,
    // CHECK-SAME:                        storage_element_table=tensor<1x1x6x6xi32, {order = #NHWC}>,
    // CHECK-SAME:                        #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:                                           scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:                                           nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 6, 6]>>

    // CHECK-NOT:  VPU.NCE.Interpolate

    // CHECK:      [[CONV_OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT_SPARSE]], [[CST_WEIGHTS]], [[CST_WEIGHTS_TABLE]]) {
    // CHECK-SAME:     pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:     ppe = #VPU.PPEStub<>,
    // CHECK-SAME:     rawFilterShape = [32, 16, 1, 1],
    // CHECK-SAME:     strides = [1, 1]
    // CHECK-SAME: } -> tensor<1x32x6x6xf16, {order = #NHWC}>
    // CHECK:      return [[CONV_OUTPUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: func.func @DoNotFuseInterpolateBilinearWithConv([[INPUT_DATA:%.+]]: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x32x6x6xf16, {order = #NHWC}> {
func.func @DoNotFuseInterpolateBilinearWithConv(%arg0: tensor<1x16x3x3xf16, {order = #NHWC}>) -> tensor<1x32x6x6xf16, {order = #NHWC}> {
    %interp_weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<0.25> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    %interp_weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %interp_sparsity_map = const.Declare tensor<1x16x7x7xi1> = dense<1> : tensor<1x16x7x7xi1>

    %interp_storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 3, 3],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 7, 7]>
    } -> tensor<1x1x7x7xi32, {order = #NHWC}>

    %interp_input = VPU.GroupSparseTensor(%arg0, %interp_sparsity_map, %interp_storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 7, 7]>
    } -> !VPU.SparseTensor<data=tensor<1x16x3x3xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x7x7xi1>,
                           storage_element_table=tensor<1x1x7x7xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 7, 7]>>

    %interp_output = VPU.NCE.Interpolate(%interp_input, %interp_weights, %interp_weights_table) {
        strides = [1, 1],
        rawFilterShape = [16, 16, 2, 2],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x6x6xf16, {order = #NHWC}>

    %conv_weights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv_weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %conv_output = VPU.NCE.Convolution(%interp_output, %conv_weights, %conv_weights_table) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 16, 1, 1],
            strides = [1, 1]
        } -> tensor<1x32x6x6xf16, {order = #NHWC}>

    return %conv_output : tensor<1x32x6x6xf16, {order = #NHWC}>

    // CHECK-DAG:  [[CONV_WEIGHTS:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:  [[CONV_WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-DAG:  [[INTERP_WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}>
    // CHECK-DAG:  [[INTERP_WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:  [[INTERP_SPARSITY_MAP:%.+]] = const.Declare tensor<1x16x7x7xi1>

    // CHECK:      [[INTERP_SE_TABLE:%.+]] = VPU.StorageElementTable
    // CHECK:      [[INTERP_INPUT:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INTERP_SPARSITY_MAP]], [[INTERP_SE_TABLE]])

    // CHECK:      [[INTERP_OUTPUT:%.+]] = VPU.NCE.Interpolate([[INTERP_INPUT]], [[INTERP_WEIGHTS]], [[INTERP_WEIGHTS_TABLE]])

    // CHECK:      [[CONV_OUTPUT:%.+]] = VPU.NCE.Convolution([[INTERP_OUTPUT]], [[CONV_WEIGHTS]], [[CONV_WEIGHTS_TABLE]])
    // CHECK:      return [[CONV_OUTPUT]]
}
