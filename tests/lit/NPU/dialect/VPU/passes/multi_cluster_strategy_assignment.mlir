//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --multi-cluster-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddAssignedSOHOverlapped
func.func @EltwiseAddAssignedSOHOverlapped(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>, %arg1: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) { op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<> } :
         tensor<1x32x112x112xf16, {order = #NHWC}>, tensor<1x32x112x112xf16, {order = #NHWC}>
         -> tensor<1x32x112x112xf16, {order = #NHWC}>
    return %0: tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        [[VAL0:%.*]] = VPU.NCE.Eltwise(%arg0, %arg1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x112x112xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOHForLargeLayer
func.func @ConvAssignedSOHForLargeLayer(%arg0: tensor<1x64x608x608xf16, {order = #NHWC}>) -> tensor<1x80x608x608xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]} -> tensor<1x80x608x608xf16, {order = #NHWC}>
    return %0 : tensor<1x80x608x608xf16, {order = #NHWC}>

    //CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    //CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    //CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    //CHECK-SAME:    {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [80, 64, 3, 3], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x80x608x608xf16, {order = #NHWC}>

    //CHECK:        return [[VAL0]] : tensor<1x80x608x608xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @InterpolateNearestAssignedClustering(%input_data: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 1, 1],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateBilinearAssignedClustering
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>)
func.func @InterpolateBilinearAssignedClustering(%input_data: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x3x3xi1> = dense<1> : tensor<1x16x3x3xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 3, 3]>
    } -> tensor<1x1x3x3xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%input_data, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 3, 3]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x3x3xi1>,
                           storage_element_table=tensor<1x1x3x3xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <BILINEAR>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], offsets = [0, 0, 0, 0], sizes = [1, 16, 3, 3]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x3x3xi1> = dense<true> : tensor<1x16x3x3xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor([[INPUT_DATA]], [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [16, 16, 2, 2],
    // CHECK-SAME:      scales_attr = [2, 2],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x2x2xf16, {order = #NHWC}>
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOKForSmallHW
// CHECK-SAME:    ([[INPUT_DATA:%.+]]: tensor<1x1024x4x4xf16, {order = #NHWC}>)
func.func @ConvAssignedSOKForSmallHW(%input_data: tensor<1x1024x4x4xf16, {order = #NHWC}>) -> tensor<1x2048x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<2048x1024x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<2048x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input_data, %weights, %weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [2048, 1024, 1, 1], strides = [2, 2]
    } -> tensor<1x2048x2x2xf16, {order = #NHWC}>

    return %conv : tensor<1x2048x2x2xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS:%.+]] = const.Declare tensor<2048x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<2048x1024x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:        [[WEIGHTS_TBL:%.+]] = const.Declare tensor<2048x1x1x4xsi32> = dense<1> : tensor<2048x1x1x4xsi32>
    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_DATA]], [[WEIGHTS]], [[WEIGHTS_TBL]])
    // CHECK-SAME:        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:        ppe = #VPU.PPEStub<>,
    // CHECK-SAME:        rawFilterShape = [2048, 1024, 1, 1], strides = [2, 2]} -> tensor<1x2048x2x2xf16, {order = #NHWC}>
    // CHECK:        return [[CONV]] : tensor<1x2048x2x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL: @EltwiseAssignedSOHWithOddWidthAndSmallHeight
func.func @EltwiseAssignedSOHWithOddWidthAndSmallHeight(%arg0: tensor<1x16x4x331776xf16, {order = #NHWC}>) -> tensor<1x16x4x16186xf16, {order = #NHWC}> {
    %eltwise1_input2 = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]
    %eltwise2_input2 = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.0> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]

    %eltwise1_input1 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    %eltwise1 = VPU.NCE.Eltwise(%eltwise1_input1, %eltwise1_input2) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    %eltwise2_input1 = VPU.Slice %arg0 [0, 0, 0, 8093] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    %eltwise2 = VPU.NCE.Eltwise(%eltwise2_input1, %eltwise2_input2) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    %concat = VPU.Concat(%eltwise1, %eltwise2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 8093]]} : tensor<1x16x4x8093xf16, {order = #NHWC}>, tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x16186xf16, {order = #NHWC}>
    return %concat : tensor<1x16x4x16186xf16, {order = #NHWC}>

    // CHECK-DAG:    [[ELTWISE1_INPUT2:%.*]] = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:    [[ELTWISE2_INPUT2:%.*]] = const.Declare tensor<1x16x4x8093xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x16x4x8093xf16>, [#const.Reorder<#NHWC>]

    // CHECK:    [[ELTWISE1_INPUT1:%.*]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    // CHECK:    [[ELTWISE1:%.*]] = VPU.NCE.Eltwise([[ELTWISE1_INPUT1]], [[ELTWISE1_INPUT2]]) {
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:  op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:  ppe = #VPU.PPEStub<>}
    // CHECK-SAME:  -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    // CHECK:    [[ELTWISE2_INPUT1:%.*]] = VPU.Slice %arg0 [0, 0, 0, 8093] [1, 16, 4, 8093] : tensor<1x16x4x331776xf16, {order = #NHWC}> to tensor<1x16x4x8093xf16, {order = #NHWC}>
    // CHECK:    [[ELTWISE2:%.*]] = VPU.NCE.Eltwise([[ELTWISE2_INPUT1]], [[ELTWISE2_INPUT2]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:  op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:  ppe = #VPU.PPEStub<>}
    // CHECK-SAME:  -> tensor<1x16x4x8093xf16, {order = #NHWC}>

    // CHECK:    [[CONCAT:%.*]] = VPU.Concat([[ELTWISE1]], [[ELTWISE2]]) {
    // CHECK:      static_offsets = [
    // CHECK-SAME:    [0, 0, 0, 0], [0, 0, 0, 8093]
    // CHECK:    ]} : tensor<1x16x4x8093xf16, {order = #NHWC}>, tensor<1x16x4x8093xf16, {order = #NHWC}> -> tensor<1x16x4x16186xf16, {order = #NHWC}>
    // return    [[CONCAT]] : tensor<1x16x4x16186xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1327104 bytes of @CMX_NN_FragmentationAware
    IE.MemoryResource 1474560 bytes of @CMX_NN
}

// CHECK-LABEL: @ConvAssignedSOKInCaseOfINFCost
// CHECK-SAME:    ([[INPUT:%.*]]: tensor<1x4096x1x1xf16, {order = #NHWC}>)
func.func @ConvAssignedSOKInCaseOfINFCost(%arg0: tensor<1x4096x1x1xf16, {order = #NHWC}>) -> tensor<1x5504x1x1xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5504x4096x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<5504x4096x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<5504x1x1x4xsi32> = dense<1> : tensor<11008x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [5504, 1, 1, 4]>]

    %conv = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [5504, 4096, 1, 1], strides = [1, 1]
    } -> tensor<1x5504x1x1xf16, {order = #NHWC}>

    return %conv : tensor<1x5504x1x1xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<5504x4096x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<5504x4096x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<5504x1x1x4xsi32> = dense<1> : tensor<11008x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [5504, 1, 1, 4]>]

    // CHECK:       [[CONV:%.*]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]]) {
    // CHECK-NOT:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      rawFilterShape = [5504, 4096, 1, 1],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      } -> tensor<1x5504x1x1xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x5504x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL: @Accumulate
// CHECK-SAME: ([[LHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS:%arg[0-9]]]: tensor<1x64x16x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[LHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:  [[RHS_SCALES:%arg[0-9]]]: tensor<1x64x1x1xf16, {order = #NHWC}>)
func.func @Accumulate(
    %LHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %RHS: tensor<1x64x16x1xf16, {order = #NHWC}>,
    %LHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>,
    %RHS_SCALES: tensor<1x64x1x1xf16, {order = #NHWC}>
) -> tensor<1x64x16x1xf16, {order = #NHWC}> {
    %ACCUMULATE = VPU.Accumulate(%LHS, %RHS, %LHS_SCALES, %RHS_SCALES) :
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x16x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>,
        tensor<1x64x1x1xf16, {order = #NHWC}>
            -> tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   [[ACCUMULATE:%.*]] = VPU.Accumulate([[LHS]], [[RHS]], [[LHS_SCALES]], [[RHS_SCALES]]) {
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  }

    return %ACCUMULATE : tensor<1x64x16x1xf16, {order = #NHWC}>

    // CHECK:   return [[ACCUMULATE]] : tensor<1x64x16x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1474560 bytes of @CMX_NN
}

// CHECK-LABEL: @MVN_PermuteCast_DepthConv_ReShape_Subgraph
func.func @MVN_PermuteCast_DepthConv_ReShape_Subgraph(%arg0: tensor<1x77x768x1xf16>) -> tensor<1x768x77x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> {
    %cst = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
    %cst_0 = const.Declare tensor<768x16x1x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> = dense<1.0> : tensor<1x1x768xf32>, [#const.Reshape<[768, 1, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[768, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>]
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 9.9999997473787516E-6 : f64, normalize_variance = true} : tensor<1x77x768x1xf16> -> tensor<1x77x768x1xf16>
    %1 = VPU.PermuteCast(%0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x77x768x1xf16> -> tensor<1x768x77x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 768, 11, 7]} : tensor<1x768x77x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x768x11x7xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %3 = VPU.NCE.DepthConvolution(%2, %cst_0, %cst) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [768, 1, 1, 1], strides = [1, 1]} -> tensor<1x768x11x7xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
    %4 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [2, 3]], shape_value = [1, 768, 77, 1]} : tensor<1x768x11x7xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x768x77x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> loc(fused<{name = "output", type = "Add"}>["output"])
    return %4 : tensor<1x768x77x1xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>

    // CHECK:       [[MVN:%.*]] = VPU.MVN
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:       [[PERMUTE_CAST:%.*]] = VPU.PermuteCast
    // CHECK:       [[AFFINE_RESHAPE:%.*]] = VPU.AffineReshape
    // CHECK:       [[NCE:%.*]] = VPU.NCE.DepthConvolution
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK:       [[AFFINE_RESHAPE_1:%.*]] = VPU.AffineReshape
    // CHECK:       return [[AFFINE_RESHAPE_1]] : tensor<1x768x77x1xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @FakeQuantizeAssignedSplitOverHeightInParamPerAxis
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x1x384x640xf16>)
func.func @FakeQuantizeAssignedSplitOverHeightInParamPerAxis(%arg0: tensor<1x1x384x640xf16>) -> tensor<1x1x384x640xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %outHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x1x384x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x384x640xf16>
    return %fq : tensor<1x1x384x640xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:    tensor<1x1x384x640xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x384x640xf16>
    //CHECK:   return [[FQ]] : tensor<1x1x384x640xf16>
}

// -----

// CHECK-LABEL: @FakeQuantizeAssignedSplitOverKernelOutParamPerAxis
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x128x1x512xf16>)
func.func @FakeQuantizeAssignedSplitOverKernelOutParamPerAxis(%arg0: tensor<1x128x1x512xf16>) -> tensor<1x128x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    %inHigh = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    %outLow = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    %outHigh = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    return %fq : tensor<1x128x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<-1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+01> : tensor<1x1x1x1xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<-1.000000e+01> : tensor<1x128x1x1xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x128x1x1xf16> = dense<1.000000e+01> : tensor<1x128x1x1xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:   tensor<1x128x1x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x128x1x1xf16>, tensor<1x128x1x1xf16> -> tensor<1x128x1x512xf16>
    //CHECK:   return [[FQ]] : tensor<1x128x1x512xf16>
}

// -----

// CHECK-LABEL: @FakeQuantizeAssignedClustering
// CHECK-SAME:  ([[INPUT_DATA:%.+]]: tensor<1x1x1x512xf16>)
func.func @FakeQuantizeAssignedClustering(%arg0: tensor<1x1x1x512xf16>) -> tensor<1x1x1x512xf16> {
    %inLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %inHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    %outLow = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    %outHigh = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    %fq = VPU.FakeQuantize(%arg0, %inLow, %inHigh, %outLow, %outHigh) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64} : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    return %fq : tensor<1x1x1x512xf16>

    //CHECK-DAG: [[IN_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[IN_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_LOW:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<-1.000000e+01> : tensor<1x1x1x512xf16>
    //CHECK-DAG: [[OUT_HIGH:%.+]] = const.Declare tensor<1x1x1x512xf16> = dense<1.000000e+01> : tensor<1x1x1x512xf16>

    //CHECK: [[FQ:%.+]] = VPU.FakeQuantize([[INPUT_DATA]], [[IN_LOW]], [[IN_HIGH]], [[OUT_LOW]], [[OUT_HIGH]])
    //CHECK-SAME:         {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>, levels = 256 : i64,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:   tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>
    //CHECK:   return [[FQ]] : tensor<1x1x1x512xf16>
}

// -----

// CHECK-LABEL: @PadAssignedSplitOverHeight
func.func @PadAssignedSplitOverHeight(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x18x20x60xf16> {

    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]} : tensor<1x16x20x50xf16> -> tensor<1x18x20x60xf16>
    return %0 : tensor<1x18x20x60xf16>

    //CHECK:   [[PAD:%.+]] = VPU.Pad({{[^:]+}}) {
    //CHECK-SAME        mode = #IE.pad_mode<EDGE>,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    //CHECK-SAME:       pad_value_attr = 0.000000e+00 : f64,
    //CHECK-SAME:       pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 2, 0, 10]}
    //CHECK-SAME:       tensor<1x16x20x50xf16> -> tensor<1x18x20x60xf16>
    //CHECK:   return [[PAD]] : tensor<1x18x20x60xf16>
}

// -----

// CHECK-LABEL: @PadAssignedSplitOverKernel
func.func @PadAssignedSplitOverKernel(%arg0: tensor<1x16x20x50xf16>) -> tensor<1x16x33x60xf16> {

    %0 = VPU.Pad(%arg0) {mode = #IE.pad_mode<EDGE>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 0, 13, 0]} : tensor<1x16x20x50xf16> -> tensor<1x16x33x60xf16>
    return %0 : tensor<1x16x33x60xf16>

    //CHECK:   [[PAD:%.+]] = VPU.Pad({{[^:]+}}) {
    //CHECK-SAME        mode = #IE.pad_mode<EDGE>,
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    //CHECK-SAME:       pad_value_attr = 0.000000e+00 : f64,
    //CHECK-SAME:       pads_begin_attr = [0, 0, 0, 10], pads_end_attr = [0, 0, 13, 0]}
    //CHECK-SAME:       tensor<1x16x20x50xf16> -> tensor<1x16x33x60xf16>
    //CHECK:   return [[PAD]] : tensor<1x16x33x60xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Mvn1NormAssignedSOH
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x256x256x256xf16, {order = #NHWC}>, [[MEAN_VAR:%.+]]: tensor<1x256x1x2xf16, {order = #NHWC}>)
func.func @Mvn1NormAssignedSOH(%arg0: tensor<1x256x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x256x1x2xf16, {order = #NHWC}>) -> tensor<1x256x256x256xf16, {order = #NHWC}> {
   %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x1x2xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
   return %0 : tensor<1x256x256x256xf16, {order = #NHWC}>

   // CHECK:       [[OUT:%.*]] = VPU.MVN1Normalize([[INPUT]], [[MEAN_VAR]])
   // CHECK-SAME :     {across_channels = false, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, normalize_variance = true} :
   // CHECK-SAME :     tensor<1x256x256x256xf16, {order = #NHWC}>, tensor<1x256x1x2xf16, {order = #NHWC}> -> tensor<1x256x256x256xf16, {order = #NHWC}>
   // CHECK:       return [[OUT]] : tensor<1x256x256x256xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SelectAssignedSplitOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x40x40xf16>, [[INPUT0:%.+]]: tensor<1x1x40x40xf16>)
func.func @SelectAssignedSplitOverHeight(%arg0: tensor<1x1x40x40xf16>, %arg1: tensor<1x1x40x40xf16>) -> tensor<1x1x40x40xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x1x40x40xf16> -> tensor<1x1x40x40xf16>
    return %0 : tensor<1x1x40x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<f16>, [#const.Reshape<[1]>, #const.Reshape<[1, 1, 1, 1]>]
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK-SAME:           } : tensor<1x1x40x40xf16>, tensor<1x1x1x1xf16>, tensor<1x1x40x40xf16> -> tensor<1x1x40x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x1x40x40xf16>
}

// -----

// CHECK-LABEL: @SelectAssignedSplitOverKernel
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x1x40xf16>, [[INPUT0:%.+]]: tensor<1x32x1x40xf16>)
func.func @SelectAssignedSplitOverKernel(%arg0: tensor<1x32x1x40xf16>, %arg1: tensor<1x32x1x40xf16>) -> tensor<1x32x1x40xf16> {
    %cst = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x40xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x40xf16> -> tensor<1x32x1x40xf16>
    return %0 : tensor<1x32x1x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x32x1x1xf16> = dense<0.000000e+00> : tensor<1x32x1x1xf16>
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK-SAME:           } : tensor<1x32x1x40xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x40xf16> -> tensor<1x32x1x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x32x1x40xf16>
}

// -----

// CHECK-LABEL: @SelectAssignedSplitOverWidth
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x1x1x40xf16>, [[INPUT0:%.+]]: tensor<1x1x1x40xf16>)
func.func @SelectAssignedSplitOverWidth(%arg0: tensor<1x1x1x40xf16>, %arg1: tensor<1x1x1x40xf16>) -> tensor<1x1x1x40xf16> {
    %cst = const.Declare tensor<1x1x1x40xf16> = dense<0.000000e+00> : tensor<1x1x1x40xf16>
    %0 = VPU.Select(%arg0, %cst, %arg1) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x1x1x40xf16>
    return %0 : tensor<1x1x1x40xf16>

    //CHECK-DAG:    [[INPUT1:%.+]] = const.Declare tensor<1x1x1x40xf16> = dense<0.000000e+00> : tensor<1x1x1x40xf16>
    //CHECK:        [[SELECT:%.+]] = VPU.Select([[INPUT]], [[INPUT1]], [[INPUT0]]) {
    //CHECK-SAME:           auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
    //CHECK-SAME:           multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>
    //CHECK-SAME:           } : tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16>, tensor<1x1x1x40xf16> -> tensor<1x1x1x40xf16>
    //CHECK:        return [[SELECT]] : tensor<1x1x1x40xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CeilingAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x16x1x513xf16, {order = #NCHW}>
func.func @CeilingAssignedSplitOverKernel(%arg0: tensor<1x16x1x513xf16, {order = #NCHW}>) -> tensor<1x16x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Ceiling(%arg0) : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x16x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.+]] = VPU.Ceiling([[INPUT_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x16x1x513xf16, {order = #NCHW}> -> tensor<1x16x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x16x1x513xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CeilingAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x16x512xf16, {order = #NCHW}>
func.func @CeilingAssignedSplitOverHeight(%arg0: tensor<1x1x16x512xf16, {order = #NCHW}>) -> tensor<1x1x16x512xf16, {order = #NCHW}> {

    %0 = VPU.Ceiling(%arg0) : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x16x512xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.+]] = VPU.Ceiling([[INPUT_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1x16x512xf16, {order = #NCHW}> -> tensor<1x1x16x512xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x16x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL:   @CeilingAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1x513xf16, {order = #NCHW}>
func.func @CeilingAssignedClustering(%arg0: tensor<1x1x1x513xf16, {order = #NCHW}>) -> tensor<1x1x1x513xf16, {order = #NCHW}> {

    %0 = VPU.Ceiling(%arg0) : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x513xf16, {order = #NCHW}>

    //CHECK:   [[Result:%.+]] = VPU.Ceiling([[INPUT_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x1x1x513xf16, {order = #NCHW}> -> tensor<1x1x1x513xf16, {order = #NCHW}>
    //CHECK:   return [[Result]] : tensor<1x1x1x513xf16, {order = #NCHW}>
}

// -----

// CHECK-LABEL:   @LSTMGatesAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x100x2048xf16>, [[INPUT_1:%.+]]: tensor<1x1x100x512xf16>
func.func @LSTMGatesAssignedSplitOverHeight(%arg0: tensor<1x1x100x2048xf16>, %arg1: tensor<1x1x100x512xf16>) -> (tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1) : tensor<1x1x100x2048xf16>, tensor<1x1x100x512xf16> -> tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>

    return %0, %1 : tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>

    //CHECK:   [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = VPU.LSTMGates([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x100x2048xf16>, tensor<1x1x100x512xf16> -> tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>
    //CHECK:   return [[LSTMGATES_0]], [[LSTMGATES_1]] : tensor<1x1x100x512xf16>, tensor<1x1x100x512xf16>
}

// -----

// CHECK-LABEL:   @LSTMGatesAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1x2048xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x512xf16>
func.func @LSTMGatesAssignedClustering(%arg0: tensor<1x1x1x2048xf16>, %arg1: tensor<1x1x1x512xf16>) -> (tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1) : tensor<1x1x1x2048xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>

    return %0, %1 : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>

    //CHECK:   [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = VPU.LSTMGates([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<1x1x1x2048xf16>, tensor<1x1x1x512xf16> -> tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>
    //CHECK:   return [[LSTMGATES_0]], [[LSTMGATES_1]] : tensor<1x1x1x512xf16>, tensor<1x1x1x512xf16>
}

// -----

// CHECK-LABEL:   @AndAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<1x106x1x1xf16>
func.func @AndAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>, %arg1: tensor<1x106x1x1xf16>) -> tensor<1x106x1x256xf16> {

    %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>

    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[And:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    //CHECK-SAME:       tensor<1x106x1x256xf16>, tensor<1x106x1x1xf16> -> tensor<1x106x1x256xf16>
    //CHECK:   return [[And]] : tensor<1x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @AndAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x256x256xf16>, [[INPUT_1:%.+]]: tensor<1x1x1x1xf16>
func.func @AndAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x256x256xf16> {

    %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>

    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[And:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
    //CHECK-SAME:       tensor<1x1x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x1x256x256xf16>
    //CHECK:   return [[And]] : tensor<1x1x256x256xf16>
}

// -----

// CHECK-LABEL:   @AndAssignedClustering
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<2x106x1x256xf16>, [[INPUT_1:%.+]]: tensor<2x106x1x1xf16>
func.func @AndAssignedClustering(%arg0: tensor<2x106x1x256xf16>, %arg1: tensor<2x106x1x1xf16>) -> tensor<2x106x1x256xf16> {

    %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>

    return %0 : tensor<2x106x1x256xf16>

    //CHECK:   [[And:%.+]] = VPU.And([[INPUT_0]], [[INPUT_1]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    //CHECK-SAME:       tensor<2x106x1x256xf16>, tensor<2x106x1x1xf16> -> tensor<2x106x1x256xf16>
    //CHECK:   return [[And]] : tensor<2x106x1x256xf16>
}

// -----

// CHECK-LABEL:   @SinAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @SinAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Sin(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[SIN:%.+]] = VPU.Sin([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @SinAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @SinAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Sin(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[SIN:%.+]] = VPU.Sin([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @SinAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @SinAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Sin(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[SIN:%.+]] = VPU.Sin([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

// CHECK-LABEL:   @CosAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @CosAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @CosAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @CosAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @CosAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @CosAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

// CHECK-LABEL:   @ExpAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @ExpAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[EXP:%.+]] = VPU.Exp([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @ExpAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @ExpAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[EXP:%.+]] = VPU.Exp([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @ExpAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @ExpAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[EXP:%.+]] = VPU.Exp([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

// CHECK-LABEL:   @SwishAssignedSplitOverWidth
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @SwishAssignedSplitOverWidth(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64}: tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[SWISH:%.+]] = VPU.Swish([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>}
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.056323952768363203:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SoftMaxAssignedSplitOverHeightAndBigConvolutionToo
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>
func.func @SoftMaxAssignedSplitOverHeightAndBigConvolutionToo(%arg0: tensor<1x48x1024x4x!qElemType, {order = #NHWC}>) -> tensor<1x4096x1024x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<4096x48x1x1x!qElemType, {order = #NHWC}> = dense<128> : tensor<4096x48x1x1xui8, {order = #NHWC}>
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>
    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    %1 = VPU.SoftMax(%0) {axisInd = 1 : i64} : tensor<1x4096x1024x4xf16, {order = #NHWC}> -> tensor<1x4096x1024x4xf16, {order = #NHWC}>
    return %1 : tensor<1x4096x1024x4xf16, {order = #NHWC}>

    //CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]],
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    //CHECK:       [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV]])
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// CHECK-LABEL:   @GatherAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x192x4x1xf16>
func.func @GatherAssignedSplitOverKernel(%arg0: tensor<1x192x4x1xf16>) -> tensor<1x192x8x1xf16> {
    %cst = const.Declare tensor<1x8x1x1xsi32> = dense<1> : tensor<1x8x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x192x4x1xf16>, tensor<1x8x1x1xsi32> -> tensor<1x192x8x1xf16>
    return %0 : tensor<1x192x8x1xf16>

    // CHECK-DAG:   [[INDICES:%.+]] = const.Declare tensor<1x8x1x1xsi32> = dense<1> : tensor<1x8x1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @GatherAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x64x72xf16>
func.func @GatherAssignedSplitOverHeight(%arg0: tensor<1x1x64x72xf16>) -> tensor<1x1x16x72xf16> {
    %cst = const.Declare tensor<1x16x1x1xsi32> = dense<1> : tensor<1x16x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x64x72xf16>, tensor<1x16x1x1xsi32> -> tensor<1x1x16x72xf16>
    return %0 : tensor<1x1x16x72xf16>

    // CHECK-DAG:   [[INDICES:%.+]] = const.Declare tensor<1x16x1x1xsi32> = dense<1> : tensor<1x16x1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @GatherAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x64x54xf16>
func.func @GatherAssignedClustering(%arg0: tensor<1x1x64x54xf16>) -> tensor<1x1x1x54xf16> {
    %cst = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
        } : tensor<1x1x64x54xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x54xf16>
    return %0 : tensor<1x1x1x54xf16>

    // CHECK-DAG:   [[INDICES:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @ConvAssignedSplitOverHeight
func.func @ConvAssignedSplitOverHeight(%arg0: tensor<1x1504x375x4xf16, {order = #NHWC}>) -> tensor<1x64x375x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<64x1504x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x1504x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %0 = VPU.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x1504x375x4xf16, {order = #NHWC}> -> tensor<1x1504x375x4xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst, %cst_1) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1504, 1, 1], strides = [1, 1]} -> tensor<1x64x375x4xf16, {order = #NHWC}>

    return %1 : tensor<1x64x375x4xf16, {order = #NHWC}>

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// CHECK-LABEL:   @MishAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x106x1x256xf16>
func.func @MishAssignedSplitOverKernel(%arg0: tensor<1x106x1x256xf16>) -> tensor<1x106x1x256xf16> {
    %0 = VPU.Mish(%arg0) : tensor<1x106x1x256xf16> -> tensor<1x106x1x256xf16>
    return %0 : tensor<1x106x1x256xf16>

    //CHECK:   [[MISH:%.+]] = VPU.Mish([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

// CHECK-LABEL:   @MishAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x256x256xf16>
func.func @MishAssignedSplitOverHeight(%arg0: tensor<1x1x256x256xf16>) -> tensor<1x1x256x256xf16> {
    %0 = VPU.Mish(%arg0) : tensor<1x1x256x256xf16> -> tensor<1x1x256x256xf16>
    return %0 : tensor<1x1x256x256xf16>

    //CHECK:   [[MISH:%.+]] = VPU.Mish([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

// CHECK-LABEL:   @MishAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x256xf16>
func.func @MishAssignedClustering(%arg0: tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16> {
    %0 = VPU.Mish(%arg0) : tensor<1x1x1x256xf16> -> tensor<1x1x1x256xf16>
    return %0 : tensor<1x1x1x256xf16>

    //CHECK:   [[MISH:%.+]] = VPU.Mish([[INPUT]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @DynamicDequantizeAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x28x4608x128x!qElemType>
// CHECK-SAME:    [[SCALE:%.+]]: tensor<1x28x4608x1xf16>
func.func @DynamicDequantizeAssignedSplitOverKernel(%arg0: tensor<1x28x4608x128x!qElemType>, %arg1: tensor<1x28x4608x1xf16>) -> tensor<1x28x4608x128xf16> {
    %0 = VPU.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16> -> tensor<1x28x4608x128xf16>
    return %0 : tensor<1x28x4608x128xf16>

    // CHECK:       [[DYNAMICDQ:%.+]] = VPU.DynamicDequantize([[INPUT]], [[SCALE]]) {dstElemType = f16,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @DynamicDequantizeAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x4608x128x!qElemType>
// CHECK-SAME:    [[SCALE:%.+]]: tensor<1x1x4608x1xf16>
func.func @DynamicDequantizeAssignedSplitOverHeight(%arg0: tensor<1x1x4608x128x!qElemType>, %arg1: tensor<1x1x4608x1xf16>) -> tensor<1x1x4608x128xf16> {
    %0 = VPU.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x1x4608x128x!qElemType>, tensor<1x1x4608x1xf16> -> tensor<1x1x4608x128xf16>
    return %0 : tensor<1x1x4608x128xf16>

    // CHECK:       [[DYNAMICDQ:%.+]] = VPU.DynamicDequantize([[INPUT]], [[SCALE]]) {dstElemType = f16,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>}
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL:   @DynamicDequantizeAssignedSplitOverWidth
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x128x!qElemType>
// CHECK-SAME:    [[SCALE:%.+]]: tensor<1x1x1x1xf16>
func.func @DynamicDequantizeAssignedSplitOverWidth(%arg0: tensor<1x1x1x128x!qElemType>, %arg1: tensor<1x1x1x1xf16>) -> tensor<1x1x1x128xf16> {
    %0 = VPU.DynamicDequantize(%arg0, %arg1) {dstElemType = f16} : tensor<1x1x1x128x!qElemType>, tensor<1x1x1x1xf16> -> tensor<1x1x1x128xf16>
    return %0 : tensor<1x1x1x128xf16>

    // CHECK:       [[DYNAMICDQ:%.+]] = VPU.DynamicDequantize([[INPUT]], [[SCALE]]) {dstElemType = f16,
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>}
}

// -----

// CHECK-LABEL:   @RMSNormAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x32x16xf16>
func.func @RMSNormAssignedSplitOverHeight(%arg: tensor<1x1x32x16xf16>) -> tensor<1x1x32x16xf16> {
    %cst_0 = const.Declare tensor<1x1x1x16xf16> = dense<[2.978500e-02, 1.403800e-02, 3.098000e-03, 1.312300e-02, 1.513700e-02, 9.399000e-03, 8.362000e-03, 0.00817899964, 1.818800e-02, 2.197300e-02, 5.249000e-03, 4.639000e-03, 4.272000e-03, 2.026400e-02, 1.348900e-02, 8.789000e-03]> : tensor<16xf32>, [#const.Reshape<[1, 1, 1, 16]>, #const.CastElemType<f16>]
    %0 = VPU.RMS(%arg, %cst_0) {epsilon = 9.9999997473787516E-6 : f64} : tensor<1x1x32x16xf16>, tensor<1x1x1x16xf16> -> tensor<1x1x32x16xf16>
    return %0 : tensor<1x1x32x16xf16>

    //CHECK:       [[GAMMA:%.+]] = const.Declare tensor<1x1x1x16xf16>
    //CHECK:       [[RMS:%.+]] = VPU.RMS([[INPUT]], [[GAMMA]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

IE.TileResource 2 of @NCE at 1.300000e+03 MHz {
    IE.MemoryResource 1000000 bytes of @CMX_NN
}

// CHECK-LABEL:   @GeluAssignedSplitOverHeight
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x2x32x16xf16>
func.func @GeluAssignedSplitOverHeight(%arg: tensor<1x2x32x16xf16>) -> tensor<1x2x32x16xf16> {
    %0 = VPU.Gelu(%arg) : tensor<1x2x32x16xf16>-> tensor<1x2x32x16xf16>
    return %0 : tensor<1x2x32x16xf16>
    //CHECK:       [[GELU:%.+]] = VPU.Gelu([[INPUT]])
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
}

// -----

// CHECK-LABEL:   @GatherElementsAssignedSplitOverKernel
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x768x512x1xf16>
func.func @GatherElementsAssignedSplitOverKernel(%arg0: tensor<1x768x512x1xf16>) -> tensor<1x768x64x1xf16> {
    %cst = const.Declare tensor<1x768x64x1xsi32> = dense<1> : tensor<1x768x64x1xsi32>
    %0 = VPU.GatherElements(%arg0, %cst) {axis = 2 : i64} : tensor<1x768x512x1xf16>, tensor<1x768x64x1xsi32> -> tensor<1x768x64x1xf16>
    return %0 : tensor<1x768x64x1xf16>

    // CHECK:       [[INDICES:%.+]] = const.Declare tensor<1x768x64x1xsi32>
    // CHECK:       [[GATHER_ELEMENTS:%.+]] = VPU.GatherElements([[INPUT]], [[INDICES]])
    // CHECK-SAME:         {axis = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    // CHECK:       return [[GATHER_ELEMENTS]] : tensor<1x768x64x1xf16>
}

// -----

// CHECK-LABEL:   @GatherElementsAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x512x1xf16>
func.func @GatherElementsAssignedClustering(%arg0: tensor<1x1x512x1xf16>) -> tensor<1x1x64x1xf16> {
    %cst = const.Declare tensor<1x1x64x1xsi32> = dense<1> : tensor<1x1x64x1xsi32>
    %0 = VPU.GatherElements(%arg0, %cst) {axis = 2 : i64} : tensor<1x1x512x1xf16>, tensor<1x1x64x1xsi32> -> tensor<1x1x64x1xf16>
    return %0 : tensor<1x1x64x1xf16>

    // CHECK:       [[INDICES:%.+]] = const.Declare tensor<1x1x64x1xsi32>
    // CHECK:       [[GATHER_ELEMENTS:%.+]] = VPU.GatherElements([[INPUT]], [[INDICES]])
    // CHECK-SAME:         {axis = 2 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
    // CHECK:        return [[GATHER_ELEMENTS]] : tensor<1x1x64x1xf16>
}

// -----

// CHECK-LABEL:   @RemoveClusteringStrategy
// CHECK-SAME:    [[INPUT0:%.+]]: tensor<1x1x1x77xf16>
// CHECK-SAME:    [[INPUT1:%.+]]: tensor<77x768xf16>
func.func @RemoveClusteringStrategy(%arg0: tensor<1x1x1x77xf16>, %arg1: tensor<77x768xf16>) -> tensor<1x768xf16> {
    %cst = const.Declare tensor<1x1x1x1232xui8> = dense<0> : tensor<1x1x1x1232xui8>
    %0 = VPU.Convert(%arg0) {dstElemType = si32} : tensor<1x1x1x77xf16> -> tensor<1x1x1x77xsi32>
    %output_values, %target_shape = VPU.TopK(%0, %cst) {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<NONE>} : tensor<1x1x1x77xsi32>, tensor<1x1x1x1232xui8> -> tensor<1x1x1x1xsi32>, tensor<1x1x1x1xsi32>
    %1 = VPU.AffineReshape(%target_shape) {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xsi32> -> tensor<1xsi32>
    %2 = VPU.Gather(%arg1, %1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<77x768xf16>, tensor<1xsi32> -> tensor<1x768xf16>
    return %2 : tensor<1x768xf16>

    // CHECK-DAG: [[TOPK_CST:%.+]] = const.Declare tensor<1x1x1x1232xui8> = dense<0> : tensor<1x1x1x1232xui8>
    // CHECK: [[CONVERT:%.+]] = VPU.Convert([[INPUT0]])
    // CHECK-SAME:  {dstElemType = si32} : tensor<1x1x1x77xf16> -> tensor<1x1x1x77xsi32>
    // CHECK-NOT:   multiClusterStrategy

    // CHECK: [[TOPK_VALUE:%.+]], [[TOPK_SHAPE:%.+]] = VPU.TopK([[CONVERT]], [[TOPK_CST]])
    // CHECK-SAME: {axis = 3 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<NONE>}
    // CHECK-NOT:  multiClusterStrategy
    // CHECK: tensor<1x1x1x77xsi32>, tensor<1x1x1x1232xui8> -> tensor<1x1x1x1xsi32>, tensor<1x1x1x1xsi32>

    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[TOPK_SHAPE]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [0]], shape_value = [1]} : tensor<1x1x1x1xsi32> -> tensor<1xsi32>
    // CHECK: [[GATHER:%.+]] = VPU.Gather([[INPUT1]], [[RESHAPE]])
    // CHECK-SAME: {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64}
    // CHECK-NOT:  multiClusterStrategy
    // CHECK: tensor<77x768xf16>, tensor<1xsi32> -> tensor<1x768xf16>
    // CHECK: return [[GATHER]] : tensor<1x768xf16>
}

// -----

// CHECK-LABEL:   @RMSNormAssignedClustering
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1x1x768xf16>
func.func @RMSNormAssignedClustering(%arg: tensor<1x1x1x768xf16>) -> tensor<1x1x1x768xf16> {
    %cst_0 = const.Declare tensor<1x1x1x768xf16> = dense<0.0> : tensor<1x1x1x768xf16>, [#const.Reshape<[1, 1, 1, 768]>, #const.CastElemType<f16>]
    %0 = VPU.RMS(%arg, %cst_0) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf16>
    return %0 : tensor<1x1x1x768xf16>

    //CHECK:       [[GAMMA:%.+]] = const.Declare tensor<1x1x1x768xf16>
    //CHECK:       [[RMS:%.+]] = VPU.RMS([[INPUT]], [[GAMMA]]) {
    //CHECK-SAME:       multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
}

// -----

// CHECK-LABEL:   @RoPEAssignedSplitOverKernel
func.func @RoPEAssignedSplitOverKernel(%arg0: tensor<1x32x1x64xf16>, %arg1: tensor<1x1x1x64xf16>, %arg2: tensor<1x1x1x64xf16>) -> tensor<1x32x1x64xf16> {
    %0 = VPU.RoPE(%arg0, %arg1, %arg2) : tensor<1x32x1x64xf16>, tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x32x1x64xf16>
    return %0 : tensor<1x32x1x64xf16>

    // CHECK:       %0 = VPU.RoPE(%arg0, %arg1, %arg2)
    // CHECK-SAME:         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    // CHECK:        return %0 : tensor<1x32x1x64xf16>
}
