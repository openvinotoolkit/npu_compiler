//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedSOH
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x128x10x10xf16, {order = #NHWC}>)
func.func @InterpolateNearestAssignedSOH(%arg0: tensor<1x128x10x10xf16, {order = #NHWC}>) -> tensor<1x128x20x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x128x20x20xi1> = dense<1> : tensor<1x128x20x20xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = [128], dataShape = [1, 128, 10, 10],
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

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [128, 128, 1, 1],
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

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      rawFilterShape = [128, 128, 1, 1],
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

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = [96], dataShape = [1, 96, 20, 20],
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

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [96, 96, 2, 2],
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

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:      rawFilterShape = [96, 96, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x96x40x40xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x96x40x40xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.010003063725490195>

// CHECK-LABEL: @TileableConvSOK
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x2560x8x8xf16, {order = #NHWC}>
func.func @TileableConvSOK(%arg0: tensor<1x2560x8x8xf16, {order = #NHWC}>) -> tensor<1x1296x8x8xf32> {
    %weights_table = const.Declare tensor<1296x1x1x4xsi32> = dense<0> : tensor<1296x1x1x4xsi32>
    %weight = const.Declare tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> : tensor<1296x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]
    %dequant = VPU.Dequantize(%weight) {dstElemType = f16} : tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1296x2560x3x3xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%arg0, %dequant, %weights_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [1296, 2560, 3, 3], strides = [1, 1]} : tensor<1x2560x8x8xf16, {order = #NHWC}>, tensor<1296x2560x3x3xf16, {order = #NHWC}>, tensor<1296x1x1x4xsi32> -> tensor<1x1296x8x8xf32>
    return %conv : tensor<1x1296x8x8xf32>

    // CHECK:   [[WEIGHTS_TABLE:%.+]] =  const.Declare tensor<1296x1x1x4xsi32> = dense<0> : tensor<1296x1x1x4xsi32>
    // CHECK:   [[WEIGHTS:%.+]] =  const.Declare tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> :
    // CHECK-SAME:  tensor<1296x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]

    // CHECK:   [[DEQUANT:%.+]] =  VPU.Dequantize([[WEIGHTS]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
    // CHECK-SAME:  tensor<1296x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1296x2560x3x3xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[ARG]], [[DEQUANT]], [[WEIGHTS_TABLE]]) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:  -> tensor<1x1296x8x8xf32>
    // CHECK:       return [[CONV]] : tensor<1x1296x8x8xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.010003063725490195>

// CHECK-LABEL: @SOHConvSOKCantBeTiled
// CHECK-SAME:  ([[ARG:%.+]]: tensor<1x2560x8x8xf16, {order = #NHWC}>
func.func @SOHConvSOKCantBeTiled(%arg0: tensor<1x2560x8x8xf16, {order = #NHWC}>) -> tensor<1x1280x8x8xf32> {
    %weights_table = const.Declare tensor<1280x1x1x4xsi32> = dense<0> : tensor<1280x1x1x4xsi32>
    %weight = const.Declare tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> : tensor<1280x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]
    %dequant = VPU.Dequantize(%weight) {dstElemType = f16} : tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1280x2560x3x3xf16, {order = #NHWC}>
    %conv = VPU.NCE.Convolution(%arg0, %dequant, %weights_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [1280, 2560, 3, 3], strides = [1, 1]} : tensor<1x2560x8x8xf16, {order = #NHWC}>, tensor<1280x2560x3x3xf16, {order = #NHWC}>, tensor<1280x1x1x4xsi32> -> tensor<1x1280x8x8xf32>
    return %conv : tensor<1x1280x8x8xf32>

        // CHECK:   [[WEIGHTS_TABLE:%.+]] =  const.Declare tensor<1280x1x1x4xsi32> = dense<0> : tensor<1280x1x1x4xsi32>
    // CHECK:   [[WEIGHTS:%.+]] =  const.Declare tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> = dense<0> :
    // CHECK-SAME:  tensor<1280x2560x3x3xui8>, [#const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.MemPermute<#NHWC, #NHWC>]

    // CHECK:   [[DEQUANT:%.+]] =  VPU.Dequantize([[WEIGHTS]]) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
    // CHECK-SAME:  tensor<1280x2560x3x3x!qElemType, {order = #NHWC}> -> tensor<1280x2560x3x3xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[ARG]], [[DEQUANT]], [[WEIGHTS_TABLE]]) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  -> tensor<1x1280x8x8xf32>
    // CHECK:       return [[CONV]] : tensor<1x1280x8x8xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOHCorrectVPUNNCost
func.func @ConvAssignedSOHCorrectVPUNNCost(%arg0: tensor<1x512x375x4xf16, {order = #NHWC}>) -> tensor<1x2048x375x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2048x1x1x4xsi32> = dense<10> : tensor<2048x1x1x4xsi32>
    %cst_0 = const.Declare tensor<2048x512x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<2048x512x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0: i64>, rawFilterShape = [2048, 512, 1, 1], strides = [1, 1]} : tensor<1x512x375x4xf16, {order = #NHWC}>, tensor<2048x512x1x1xf16, {order = #NHWC}>, tensor<2048x1x1x4xsi32> -> tensor<1x2048x375x4xf16, {order = #NHWC}>
    %1 = VPU.Gelu(%0) : tensor<1x2048x375x4xf16, {order = #NHWC}> -> tensor<1x2048x375x4xf16, {order = #NHWC}>
    return %1 : tensor<1x2048x375x4xf16, {order = #NHWC}>

    // With SOK_NO_BROADCAST VPUNN strategy, SOK is chosen for conv
    // Otherwise SOH will be chosen
    //CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:        [[GELU:%.+]] = VPU.Gelu
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceEltwiseAssignedSEGSOK
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x7168x1x1xf16, {order = #NHWC}>, [[ARG1:%.+]]: tensor<1x7168x1x1xf16, {order = #NHWC}>)
// CHECK-SAME:  -> tensor<1x7168x1x1xf16, {order = #NHWC}>
func.func @NceEltwiseAssignedSEGSOK(%arg0:  tensor<1x7168x1x1xf16, {order = #NHWC}>, %arg1: tensor<1x7168x1x1xf16, {order = #NHWC}>) -> tensor<1x7168x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<MULTIPLY>,
        ppe = #VPU.PPEFp<mode = <NOOP>,
        clamp_low = -3.4028234663852886E+38 : f64,
        clamp_high = 3.4028234663852886E+38 : f64,
        scale = 1.000000e+00 : f64,
        prelu_alpha = [1.000000e+00],
        bias = 0.000000e+00 : f64,
        adder = 0.000000e+00 : f64>
    } -> tensor<1x7168x1x1xf16, {order = #NHWC}>
    return %0 : tensor<1x7168x1x1xf16, {order = #NHWC}>
    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:  op_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK-SAME:  -> tensor<1x7168x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x7168x1x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @ConcatInputHasReturnConsumer
// CHECK-SAME:  ([[ARG0:%.+]]: tensor<1x128x1x256xf16>, [[ARG1:%.+]]: tensor<1x896x1x256xf16>)
func.func @ConcatInputHasReturnConsumer(%arg0:  tensor<1x128x1x256xf16>, %arg1: tensor<1x896x1x256xf16>) -> (tensor<1x128x1x256xf16>, tensor<1x1024x1x256xf16>) {
    %cst = const.Declare tensor<1x1x1x256xf16> = dense<1.000000e+00> : tensor<256xf32>, [#const.Reshape<[1, 1, 1, 256]>, #const.CastElemType<f16>]
    %0 = VPU.RMS(%arg0, %cst) {eps = 1.0132789611816406E-6 : f64} : tensor<1x128x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x128x1x256xf16>
    %1 = VPU.Concat(%0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x1x256xf16>, tensor<1x896x1x256xf16> -> tensor<1x1024x1x256xf16>

    return %0, %1 : tensor<1x128x1x256xf16>, tensor<1x1024x1x256xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x256xf16> = dense<1.000000e+00> : tensor<256xf32>, [#const.Reshape<[1, 1, 1, 256]>, #const.CastElemType<f16>]
    // CHECK:       [[RMS:%.+]] = VPU.RMS([[ARG0]], [[CST]]) {eps = 1.0132789611816406E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x128x1x256xf16>, tensor<1x1x1x256xf16> -> tensor<1x128x1x256xf16>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[RMS]], [[ARG1]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x1x256xf16>, tensor<1x896x1x256xf16> -> tensor<1x1024x1x256xf16>

    // CHECK:       return [[RMS]], [[CONCAT]] : tensor<1x128x1x256xf16>, tensor<1x1024x1x256xf16>

}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @MultiplyInconsistentShapeSplitOverHeight
func.func @MultiplyInconsistentShapeSplitOverHeight(%arg0: tensor<1x4x256x1xf16, {order = #map}>,
            %arg1: tensor<1x1x256x2048xf16, {order = #map}>) -> tensor<1x4x256x2048xf16, {order = #map}> {

    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x4x256x1xf16, {order = #map}>,
                tensor<1x1x256x2048xf16, {order = #map}> -> tensor<1x4x256x2048xf16, {order = #map}>

    return %0 : tensor<1x4x256x2048xf16, {order = #map}>

    //CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x256x1xf16, {order = #map}>, tensor<1x1x256x2048xf16, {order = #map}> -> tensor<1x4x256x2048xf16, {order = #map}>
    //CHECK:      return [[MULTIPLY]] : tensor<1x4x256x2048xf16, {order = #map}>
}
