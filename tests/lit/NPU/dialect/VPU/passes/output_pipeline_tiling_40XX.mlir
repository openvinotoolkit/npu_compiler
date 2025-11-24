//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true num-of-dpu-groups=2" --output-pipeline-tiling %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @IncreaseNumTilesForNCEConv
func.func @IncreaseNumTilesForNCEConv(%input: tensor<1x32x1088x480xf16, {order = #NHWC}>)
            -> tensor<1x32x1088x480xf16, {order = #NHWC}> {
    %weightsData = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %weightsSM = const.Declare tensor<32x1x1x384xi1> = dense<0.0> : tensor<32x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]

    %filter = VPU.GroupSparseTensor(%weightsData, %weightsSM) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
            -> !VPU.SparseTensor<data=tensor<32x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<32x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<32xi64>, alignment = 16 : i64>>

    %weightsTBL = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %filter, %weightsTBL) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 32, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 46, 1]
    } : tensor<1x32x1088x480xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<32x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<32x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<72> : tensor<32xi64>, alignment = 16 : i64>>, tensor<32x1x1x4xsi32> -> tensor<1x32x1088x480xf16, {order = #NHWC}>

    return %conv : tensor<1x32x1088x480xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [32, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-NOT:           tilingStrategy = [1, 1, 46, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 64, 1]}
    // CHECK-SAME:      -> tensor<1x32x1088x480xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x1088x480xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotChangeTilingStrategyForVF
func.func @NotChangeTilingStrategyForVF(%input: tensor<1x32x135x240xf16, {order = #NHWC}>)
            -> tensor<1x32x135x240xf16, {order = #NHWC}> {
    %weights0 = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weightsTable0 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %weights1 = const.Declare tensor<32x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x128x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %weightsTable1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %vf = VPU.VerticalFusion (%input as %arg1: tensor<1x32x135x240xf16, {order = #NHWC}>,
                              %weights0 as %arg2: tensor<128x32x3x3xf16, {order = #NHWC}>,
                              %weightsTable0 as %arg3: tensor<128x1x1x4xsi32>,
                              %weights1 as %arg4: tensor<32x128x3x3xf16, {order = #NHWC}>,
                              %weightsTable1 as %arg5: tensor<32x1x1x4xsi32>)
            attributes {tilingStrategy = [1, 1, 11, 1]}
            -> tensor<1x32x135x240xf16, {order = #NHWC}> {
        %conv0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [128, 32, 3, 3],
            strides = [1, 1]
            } : tensor<1x32x135x240xf16, {order = #NHWC}>, tensor<128x32x3x3xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x135x240xf16, {order = #NHWC}>
        %conv1 = VPU.NCE.Convolution(%conv0, %arg4, %arg5) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 128, 3, 3],
            strides = [1, 1]
            } : tensor<1x128x135x240xf16, {order = #NHWC}>, tensor<32x128x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x135x240xf16, {order = #NHWC}>
        %add = VPU.NCE.Eltwise(%conv1, %arg1) {
            is_inplace = true,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } -> tensor<1x32x135x240xf16, {order = #NHWC}>

        VPU.Yield %add
    }

    return %vf : tensor<1x32x135x240xf16, {order = #NHWC}>

    // CHECK:       [[VF:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          attributes {tilingStrategy = [1, 1, 11, 1]}
    // CHECK-SAME:          -> tensor<1x32x135x240xf16, {order = #NHWC}> {
    // CHECK:           VPU.NCE.Convolution
    // CHECK:           VPU.NCE.Convolution
    // CHECK:           VPU.NCE.Eltwise
    // CHECK:           VPU.Yield

    // CHECK:       return [[VF]] : tensor<1x32x135x240xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.057961288152956494:128>
!qElemType1 = !quant.uniform<u8:f16, 1.1534313725490195:128>
!qElemType2 = !quant.uniform<u8:f16, 0.017328284768497244>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @TilingWithMinFragmentation
func.func @TilingWithMinFragmentation(%input: tensor<1x256x184x240x!qElemType, {order = #NHWC}>)
            -> tensor<1x128x184x240x!qElemType2, {order = #NHWC}> {
    %weights = const.Declare tensor<128x256x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<128x256x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %weightsTBL = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %weights, %weightsTBL) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [128, 256, 3, 3],
        strides = [1, 1],
        tilingStrategy = [1, 1, 13, 1]
    } : tensor<1x256x184x240x!qElemType, {order = #NHWC}>, tensor<128x256x3x3x!qElemType1, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x184x240x!qElemType2, {order = #NHWC}>

    return %conv : tensor<1x128x184x240x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution(%arg0, %cst, %cst_0) {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [128, 256, 3, 3],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 23, 1]}
    // CHECK-SAME:      -> tensor<1x128x184x240x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x128x184x240x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @AvoidExcessiveTilingForLargeConv
func.func @AvoidExcessiveTilingForLargeConv(%input: tensor<1x128x256x4xf16, {order = #NHWC}>)
            -> tensor<1x6320x256x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<6320x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<6320x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %weightsTBL = const.Declare tensor<6320x1x1x4xsi32> = dense<1> : tensor<6320x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %weights, %weightsTBL) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        opaque_ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64,
            lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [6320, 128, 1, 1], strides = [1, 1],
        tilingStrategy = [1, 5, 1, 1]
    } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<6320x128x1x1xf16, {order = #NHWC}>, tensor<6320x1x1x4xsi32> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    return %conv : tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-NOT:           tilingStrategy = [1, 79, 1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 18, 1, 1]}
    // CHECK-SAME:      ->  tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
}
