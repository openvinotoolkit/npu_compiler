//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedSOK
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x128x10x10xf16, {order = #NHWC}>)
func.func @InterpolateNearestAssignedSOK(%arg0: tensor<1x128x10x10xf16, {order = #NHWC}>) -> tensor<1x128x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x128x20x20xi1> = dense<1> : tensor<1x128x20x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [128], dataShape = [1, 128, 10, 10],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 20, 20]>
    } -> tensor<1x1x20x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 128, 20, 20]>
    } -> !VPU.SparseTensor<data=tensor<1x128x10x10xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x128x20x20xi1>,
                           storage_element_table=tensor<1x1x20x20xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 128, 20, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) rawFilterShape [128, 128, 1, 1] {

        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x128x20x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x128x20x20xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x128x20x20xi1> = dense<true> : tensor<1x128x20x20xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) rawFilterShape [128, 128, 1, 1] {
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x20x20xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x128x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedSOH
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x96x20x20xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAssignedSOH(%arg0: tensor<1x96x20x20xf16, {order = #NHWC}>) -> tensor<1x96x40x40xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<96x96x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<96x96x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x96x41x41xi1> = dense<1> : tensor<1x96x41x41xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = f16, seDepth = 1, seSize = [96], dataShape = [1, 96, 20, 20],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 96, 41, 41]>
    } -> tensor<1x1x41x41xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 96, 41, 41]>
    } -> !VPU.SparseTensor<data=tensor<1x96x20x20xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x96x41x41xi1>,
                           storage_element_table=tensor<1x1x41x41xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 96, 41, 41]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) rawFilterShape [96, 96, 2, 2] {

        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x96x40x40xf16, {order = #NHWC}>

    return %interpolate : tensor<1x96x40x40xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<96x96x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x96x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x96x41x41xi1> = dense<true> : tensor<1x96x41x41xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[ARG]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) rawFilterShape [96, 96, 2, 2] {
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x96x40x40xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SEPInterpolateODUAutopad
module @SEPInterpolateODUAutopad  {
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK:  @main([[DATA:%.+]]: tensor<1x16x256x1024xf16, {order = #NHWC}>)
func.func @main(%data: tensor<1x16x256x1024xf16, {order = #NHWC}>) -> tensor<1x4x1024x1024xf32> {
    %se_table = VPU.StorageElementTable {
            dataElemType = f16,
            dataShape = [1, 16, 256, 1024],
            seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 4.000000e+00, 1.000000e+00]>, seDepth = 1 : i64, seSize = [16]
        } -> tensor<1x1x1028x1024xi32, {order = #NHWC}>
    %input = VPU.GroupSparseTensor(%data, %se_table) {
            operandSegmentSizes = array<i32: 1, 0, 1>,
            seAttr = #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <HALF_PIXEL>,
                scale = [1.000000e+00, 1.000000e+00, 4.000000e+00, 1.000000e+00]>
        } -> !VPU.SparseTensor<data=tensor<1x16x256x1024xf16, {order = #NHWC}>,
                               storage_element_table=tensor<1x1x1028x1024xi32, {order = #NHWC}>,
                               #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <HALF_PIXEL>, scale = [1.000000e+00, 1.000000e+00, 4.000000e+00, 1.000000e+00]>>

    %weights = const.Declare tensor<4x16x5x1xf16, {order = #NHWC}> = dense<1.0> : tensor<4x16x5x1xf16>, [#const.Reorder<#NHWC>]
    %scale = const.Declare tensor<16x1x1x1xf32> = dense<1.0> : tensor<16x1x1x1xf32>
    %bias = const.Declare tensor<16x1x1x1xf32> = dense<0.0> : tensor<16x1x1x1xf32>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %scale, %bias) rawFilterShape [16, 16, 5, 1] {
        input_padding = [0, 12, 0, 0],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        operandSegmentSizes = array<i32: 1, 1, 0, 0, 1, 1, 0>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x4x1024x1024xf32>

    return %interpolate : tensor<1x4x1024x1024xf32>


    // CHECK:       [[SE_TABLE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT:%.+]] = VPU.GroupSparseTensor([[DATA]], [[SE_TABLE]])

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<4x16x5x1xf16, {order = #NHWC}>
    // CHECK-DAG:   [[SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<1.000000e+00>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<16x1x1x1xf32> = dense<0.000000e+00>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT]], [[WEIGHTS]], [[SCALE]], [[BIAS]]) rawFilterShape [16, 16, 5, 1] {
    // CHECK-SAME:      input_padding = [0, 12, 0, 0],
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x4x1024x1024xf32>
    // CHECK:       return [[OUTPUT]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOHCorrectVPUNNCost
func.func @ConvAssignedSOHCorrectVPUNNCost(%arg0: tensor<1x512x375x4xf16, {order = #NHWC}>) -> tensor<1x2048x375x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2048x1x1x4xsi32> = dense<10> : tensor<2048x1x1x4xsi32>
    %cst_0 = const.Declare tensor<2048x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<2048x512x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) rawFilterShape [2048, 512, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0: i64>,  strides = [1, 1]} : tensor<1x512x375x4xf16, {order = #NHWC}>, tensor<2048x512x1x1xf16, {order = #NHWC}>, tensor<2048x1x1x4xsi32> -> tensor<1x2048x375x4xf16, {order = #NHWC}>
    %1 = VPU.Gelu(%0) : tensor<1x2048x375x4xf16, {order = #NHWC}> -> tensor<1x2048x375x4xf16, {order = #NHWC}>
    return %1 : tensor<1x2048x375x4xf16, {order = #NHWC}>

    //CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:        [[GELU:%.+]] = VPU.Gelu
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOHMaxPoolAsConvScaleAssignedClustering
// CHECK-SAME:  ([[POOL_INPUT:%.+]]: tensor<1x16x16x8xf16, {order = #NHWC}>, [[CONV_INPUT:%.+]]: tensor<1x128x1x1xf16, {order = #NHWC}>)
func.func @SOHMaxPoolAsConvScaleAssignedClustering(%pool_input: tensor<1x16x16x8xf16, {order = #NHWC}>, %conv_input: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1024x1x1x1xf32> = dense<0.000000e+00> : tensor<1024x1x1x1xf32>
    %pool = VPU.NCE.MaxPool(%pool_input) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<1x16x16x8xf32, {order = #NHWC}>
    %layout = VPU.LayoutCast(%pool) {dst_order = #NCHW} : tensor<1x16x16x8xf32, {order = #NHWC}> -> tensor<1x16x16x8xf32>
    %scale_table = VPU.AffineReshape(%layout) {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [2048, 1, 1, 1]} : tensor<1x16x16x8xf32> -> tensor<2048x1x1x1xf32>
    %scale = VPU.Slice %scale_table [0, 0, 0, 0] [1024, 1, 1, 1] : tensor<2048x1x1x1xf32> to tensor<1024x1x1x1xf32>
    %conv = VPU.NCE.Convolution(%conv_input, %weights, %scale, %bias) rawFilterShape [1024, 128, 1, 1] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x1xf32>, tensor<1024x1x1x1xf32> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %conv : tensor<1x1024x1x1xf16, {order = #NHWC}>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}>
    // CHECK:      [[BIAS:%.+]] = const.Declare tensor<1024x1x1x1xf32>
    // CHECK:      [[POOL:%.+]] = VPU.NCE.MaxPool([[POOL_INPUT]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:      [[SCALE:%.+]] = VPU.Slice
    // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[WEIGHTS]], [[SCALE]], [[BIAS]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:      return [[CONV]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOKMaxPoolAsConvScaleAssignedClustering
// CHECK-SAME:  ([[POOL_INPUT:%.+]]: tensor<1x256x4x8xf16, {order = #NHWC}>, [[CONV_INPUT:%.+]]: tensor<1x128x4x2xf16, {order = #NHWC}>)
func.func @SOKMaxPoolAsConvScaleAssignedClustering(%pool_input: tensor<1x256x4x8xf16, {order = #NHWC}>, %conv_input: tensor<1x128x4x2xf16, {order = #NHWC}>) -> tensor<1x8192x4x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<8192x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<8192x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<8192x1x1x1xf32> = dense<0.000000e+00> : tensor<8192x1x1x1xf32>
    %pool = VPU.NCE.MaxPool(%pool_input) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<1x256x4x8xf32, {order = #NHWC}>
    %layout = VPU.LayoutCast(%pool) {dst_order = #NCHW} : tensor<1x256x4x8xf32, {order = #NHWC}> -> tensor<1x256x4x8xf32>
    %scale_table = VPU.AffineReshape(%layout) {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [8192, 1, 1, 1]} : tensor<1x256x4x8xf32> -> tensor<8192x1x1x1xf32>
    %conv = VPU.NCE.Convolution(%conv_input, %weights, %scale_table, %bias) rawFilterShape [8192, 128, 1, 1] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x128x4x2xf16, {order = #NHWC}>, tensor<8192x128x1x1xf16, {order = #NHWC}>, tensor<8192x1x1x1xf32>, tensor<8192x1x1x1xf32> -> tensor<1x8192x4x2xf16, {order = #NHWC}>
    return %conv : tensor<1x8192x4x2xf16, {order = #NHWC}>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<8192x128x1x1xf16, {order = #NHWC}>
    // CHECK:      [[BIAS:%.+]] = const.Declare tensor<8192x1x1x1xf32>
    // CHECK:      [[POOL:%.+]] = VPU.NCE.MaxPool([[POOL_INPUT]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:      [[SCALE_TABLE:%.+]] = VPU.AffineReshape
    // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:      return [[CONV]] : tensor<1x8192x4x2xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOHMultiplyAsConvScaleAssignedClustering
// CHECK-SAME:  ([[ELTWISE_INPUT0:%[^:]+]]: tensor<1x16x16x8xf16, {order = #NHWC}>, [[ELTWISE_INPUT1:%[^:]+]]: tensor<1x16x16x8xf16, {order = #NHWC}>, [[CONV_INPUT:%[^:]+]]: tensor<1x128x1x1xf16, {order = #NHWC}>)
func.func @SOHMultiplyAsConvScaleAssignedClustering(%eltwise_input0: tensor<1x16x16x8xf16, {order = #NHWC}>, %eltwise_input1: tensor<1x16x16x8xf16, {order = #NHWC}>, %conv_input: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x1024x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1024x1x1x1xf32> = dense<0.000000e+00> : tensor<1024x1x1x1xf32>
    %eltwise = VPU.NCE.Eltwise(%eltwise_input0, %eltwise_input1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEStub<>} -> tensor<1x16x16x8xf32, {order = #NHWC}>
    %layout = VPU.LayoutCast(%eltwise) {dst_order = #NCHW} : tensor<1x16x16x8xf32, {order = #NHWC}> -> tensor<1x16x16x8xf32>
    %scale_table = VPU.AffineReshape(%layout) {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [2048, 1, 1, 1]} : tensor<1x16x16x8xf32> -> tensor<2048x1x1x1xf32>
    %scale = VPU.Slice %scale_table [0, 0, 0, 0] [1024, 1, 1, 1] : tensor<2048x1x1x1xf32> to tensor<1024x1x1x1xf32>
    %conv = VPU.NCE.Convolution(%conv_input, %weights, %scale, %bias) rawFilterShape [1024, 128, 1, 1] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x128x1x1xf16, {order = #NHWC}>, tensor<1024x128x1x1xf16, {order = #NHWC}>, tensor<1024x1x1x1xf32>, tensor<1024x1x1x1xf32> -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    return %conv : tensor<1x1024x1x1xf16, {order = #NHWC}>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<1024x128x1x1xf16, {order = #NHWC}>
    // CHECK:      [[BIAS:%.+]] = const.Declare tensor<1024x1x1x1xf32>
    // CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT0]], [[ELTWISE_INPUT1]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK-SAME:     op_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK:      [[SCALE:%.+]] = VPU.Slice
    // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[WEIGHTS]], [[SCALE]], [[BIAS]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:      return [[CONV]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SOKMultiplyAsConvScaleAssignedClustering
// CHECK-SAME:  ([[ELTWISE_INPUT0:%[^:]+]]: tensor<1x256x4x8xf16, {order = #NHWC}>, [[ELTWISE_INPUT1:%[^:]+]]: tensor<1x256x4x8xf16, {order = #NHWC}>, [[CONV_INPUT:%[^:]+]]: tensor<1x128x4x2xf16, {order = #NHWC}>)
func.func @SOKMultiplyAsConvScaleAssignedClustering(%eltwise_input0: tensor<1x256x4x8xf16, {order = #NHWC}>, %eltwise_input1: tensor<1x256x4x8xf16, {order = #NHWC}>, %conv_input: tensor<1x128x4x2xf16, {order = #NHWC}>) -> tensor<1x8192x4x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<8192x128x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<8192x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<8192x1x1x1xf32> = dense<0.000000e+00> : tensor<8192x1x1x1xf32>
    %eltwise = VPU.NCE.Eltwise(%eltwise_input0, %eltwise_input1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, op_type = #VPU.eltwise_type<MULTIPLY>, ppe = #VPU.PPEStub<>} -> tensor<1x256x4x8xf32, {order = #NHWC}>
    %layout = VPU.LayoutCast(%eltwise) {dst_order = #NCHW} : tensor<1x256x4x8xf32, {order = #NHWC}> -> tensor<1x256x4x8xf32>
    %scale_table = VPU.AffineReshape(%layout) {dim_mapping = [[0], [0], [0], [0, 1, 2, 3]], shape_value = [8192, 1, 1, 1]} : tensor<1x256x4x8xf32> -> tensor<8192x1x1x1xf32>
    %conv = VPU.NCE.Convolution(%conv_input, %weights, %scale_table, %bias) rawFilterShape [8192, 128, 1, 1] {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1], operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>} : tensor<1x128x4x2xf16, {order = #NHWC}>, tensor<8192x128x1x1xf16, {order = #NHWC}>, tensor<8192x1x1x1xf32>, tensor<8192x1x1x1xf32> -> tensor<1x8192x4x2xf16, {order = #NHWC}>
    return %conv : tensor<1x8192x4x2xf16, {order = #NHWC}>

    // CHECK:      [[WEIGHTS:%.+]] = const.Declare tensor<8192x128x1x1xf16, {order = #NHWC}>
    // CHECK:      [[BIAS:%.+]] = const.Declare tensor<8192x1x1x1xf32>
    // CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT0]], [[ELTWISE_INPUT1]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK-SAME:     op_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK:      [[SCALE_TABLE:%.+]] = VPU.AffineReshape
    // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[CONV_INPUT]], [[WEIGHTS]], [[SCALE_TABLE]], [[BIAS]])
    // CHECK-SAME:     multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:      return [[CONV]] : tensor<1x8192x4x2xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @GatedDeltaNetTileOverHead
func.func @GatedDeltaNetTileOverHead(%q: tensor<1x512x4x128xf16>, %k: tensor<1x512x4x128xf16>, %v: tensor<1x512x4x128xf16>,
                                     %s: tensor<1x4x128x128xf32>, %g: tensor<1x512x4xf16>, %b: tensor<1x512x4xf16>,
                                     %scratch: tensor<1x1x1x931968xui8>)
        -> (tensor<1x512x4x128xf16>, tensor<1x4x128x128xf32>) {
    %out, %state = VPU.GatedDeltaNet(%q, %k, %v, %s, %g, %b, %scratch) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x512x4x128xf16>, tensor<1x512x4x128xf16>, tensor<1x512x4x128xf16>, tensor<1x4x128x128xf32>,
        tensor<1x512x4xf16>, tensor<1x512x4xf16>, tensor<1x1x1x931968xui8> -> tensor<1x512x4x128xf16>, tensor<1x4x128x128xf32>
    return %out, %state : tensor<1x512x4x128xf16>, tensor<1x4x128x128xf32>

    // CHECK:       VPU.GatedDeltaNet
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// CHECK-LABEL: @GatedDeltaNetGQAUnevenOver4Tiles
func.func @GatedDeltaNetGQAUnevenOver4Tiles(%q: tensor<1x512x6x128xf16>, %k: tensor<1x512x6x128xf16>, %v: tensor<1x512x12x128xf16>,
                                            %s: tensor<1x12x128x128xf32>, %g: tensor<1x512x12xf16>, %b: tensor<1x512x12xf16>,
                                            %scratch: tensor<1x1x1x931968xui8>)
        -> (tensor<1x512x12x128xf16>, tensor<1x12x128x128xf32>) {
    %out, %state = VPU.GatedDeltaNet(%q, %k, %v, %s, %g, %b, %scratch) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x512x6x128xf16>, tensor<1x512x6x128xf16>, tensor<1x512x12x128xf16>, tensor<1x12x128x128xf32>,
        tensor<1x512x12xf16>, tensor<1x512x12xf16>, tensor<1x1x1x931968xui8> -> tensor<1x512x12x128xf16>, tensor<1x12x128x128xf32>
    return %out, %state : tensor<1x512x12x128xf16>, tensor<1x12x128x128xf32>

    // CHECK:       VPU.GatedDeltaNet
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// qk=4 heads are divisible by 4 tiles, so this GQA case can SplitOverHeight on NPU6010.

// CHECK-LABEL: @GatedDeltaNetGQAAlignedTo4Tiles
func.func @GatedDeltaNetGQAAlignedTo4Tiles(%q: tensor<1x512x4x128xf16>, %k: tensor<1x512x4x128xf16>, %v: tensor<1x512x8x128xf16>,
                                           %s: tensor<1x8x128x128xf32>, %g: tensor<1x512x8xf16>, %b: tensor<1x512x8xf16>,
                                           %scratch: tensor<1x1x1x931968xui8>)
        -> (tensor<1x512x8x128xf16>, tensor<1x8x128x128xf32>) {
    %out, %state = VPU.GatedDeltaNet(%q, %k, %v, %s, %g, %b, %scratch) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x512x4x128xf16>, tensor<1x512x4x128xf16>, tensor<1x512x8x128xf16>, tensor<1x8x128x128xf32>,
        tensor<1x512x8xf16>, tensor<1x512x8xf16>, tensor<1x1x1x931968xui8> -> tensor<1x512x8x128xf16>, tensor<1x8x128x128xf32>
    return %out, %state : tensor<1x512x8x128xf16>, tensor<1x8x128x128xf32>

    // CHECK:       VPU.GatedDeltaNet
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// CHECK-LABEL: @GatedDeltaNetFitsCMX
// CHECK-NOT:   multiClusterStrategy
func.func @GatedDeltaNetFitsCMX(%q: tensor<1x4x2x8xf16>, %k: tensor<1x4x2x8xf16>, %v: tensor<1x4x2x8xf16>,
                                %s: tensor<1x2x8x8xf32>, %g: tensor<1x4x2xf16>, %b: tensor<1x4x2xf16>,
                                %scratch: tensor<1x1x1x191616xui8>)
        -> (tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>) {
    %out, %state = VPU.GatedDeltaNet(%q, %k, %v, %s, %g, %b, %scratch) {
        q_l2_norm_eps = 1.000000e-03 : f64, k_l2_norm_eps = 2.000000e-03 : f64
    } : tensor<1x4x2x8xf16>, tensor<1x4x2x8xf16>, tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>,
        tensor<1x4x2xf16>, tensor<1x4x2xf16>, tensor<1x1x1x191616xui8> -> tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>
    return %out, %state : tensor<1x4x2x8xf16>, tensor<1x2x8x8xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// HKSwitch on %pool0 output (1.5 MB) would be kept in every tile while it feeds both the network output and the next MaxPool.

// CHECK-LABEL: @NoHKSwitchForLargeOutputWithMultipleUsers
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x64x1500x8xf16, {order = #NHWC}>)
func.func @NoHKSwitchForLargeOutputWithMultipleUsers(%arg0: tensor<1x64x1500x8xf16, {order = #NHWC}>)
        -> (tensor<1x8x1500x64xf16>, tensor<1x8x64x1500xf16>) {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %conv = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [64, 64, 1, 1] {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]
        } : tensor<1x64x1500x8xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x1500x8xf16, {order = #NHWC}>
    %pool0 = VPU.NCE.MaxPool(%conv) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<1x64x1500x8xf16, {order = #NWHC}>
    %out0 = VPU.PermuteCast(%pool0) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x1500x8xf16, {order = #NWHC}> -> tensor<1x8x1500x64xf16>
    %cast = VPU.PermuteCast(%pool0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x1500x8xf16, {order = #NWHC}> -> tensor<1x64x8x1500xf16, {order = #NHWC}>
    %pool1 = VPU.NCE.MaxPool(%cast) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<1x64x8x1500xf16, {order = #NHCW}>
    %out1 = VPU.PermuteCast(%pool1) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x8x1500xf16, {order = #NHCW}> -> tensor<1x8x64x1500xf16>
    return %out0, %out1 : tensor<1x8x1500x64xf16>, tensor<1x8x64x1500xf16>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[ARG]]
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       [[POOL0:%.+]] = VPU.NCE.MaxPool([[CONV]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       [[OUT0:%.+]] = VPU.PermuteCast([[POOL0]])
    // CHECK:       [[CAST:%.+]] = VPU.PermuteCast([[POOL0]])
    // CHECK:       [[POOL1:%.+]] = VPU.NCE.MaxPool([[CAST]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK:       [[OUT1:%.+]] = VPU.PermuteCast([[POOL1]])
    // CHECK:       return [[OUT0]], [[OUT1]]
}
