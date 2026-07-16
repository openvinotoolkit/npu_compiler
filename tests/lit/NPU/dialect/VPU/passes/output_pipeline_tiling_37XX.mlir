//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --output-pipeline-tiling %s | FileCheck %s
// REQUIRES: platform-NPU3720

!qElemType0 = !quant.uniform<u8:f16, 0.14634351543351715>
!qElemType1 = !quant.uniform<u8:f16, 1.000000e+00>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @IncreaseNumTilesForNCEConv
func.func @IncreaseNumTilesForNCEConv(%input: tensor<1x16x480x320x!qElemType0, {order = #NHWC}>)
            -> tensor<1x32x480x320x!qElemType0, {order = #NHWC}> {
    %weightsData = const.Declare tensor<32x16x1x1x!qElemType1, {order = #NHWC}> = dense<1> : tensor<32x16x1x1xui8, {order = #NHWC}>, [#const.CastElemType<!qElemType1>, #const.Sparsify<false>]
    %weightsSM = const.Declare tensor<32x1x1x128xi1> = dense<0> : tensor<32x16x1x1xui8, {order = #NHWC}>, [#const.CastElemType<!qElemType1>, #const.GetSparsityMap]

    %filter = VPU.GroupSparseTensor(%weightsData, %weightsSM) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
            -> !VPU.SparseTensor<data=tensor<32x16x1x1x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<32x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<32xi64>, alignment = 16 : i64>>

    %weightsTBL = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %conv = VPU.NCE.Convolution(%input, %filter, %weightsTBL) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1],
        tilingStrategy = [1, 1, 3, 1]
    } : tensor<1x16x480x320x!qElemType0, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<32x16x1x1x!qElemType1, {order = #NHWC}>, sparsity_map=tensor<32x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<32xi64>, alignment = 16 : i64>>, tensor<32x1x1x4xsi32> -> tensor<1x32x480x320x!qElemType0, {order = #NHWC}>

    return %conv : tensor<1x32x480x320x!qElemType0, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution{{.*}} rawFilterShape [32, 16, 1, 1]
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1],
    // CHECK-NOT:           tilingStrategy = [1, 1, 3, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 5, 1]}
    // CHECK-SAME:      -> tensor<1x32x480x320x!qElemType, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x480x320x!qElemType, {order = #NHWC}>
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
            attributes {tilingStrategy = [1, 1, 7, 1]}
            -> tensor<1x32x135x240xf16, {order = #NHWC}> {
        %conv0 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) rawFilterShape [128, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            
            strides = [1, 1]
            } : tensor<1x32x135x240xf16, {order = #NHWC}>, tensor<128x32x3x3xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x135x240xf16, {order = #NHWC}>
        %conv1 = VPU.NCE.Convolution(%conv0, %arg4, %arg5) rawFilterShape [32, 128, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            
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
    // CHECK-SAME:          attributes {tilingStrategy = [1, 1, 7, 1]}
    // CHECK-SAME:          -> tensor<1x32x135x240xf16, {order = #NHWC}> {
    // CHECK:           VPU.NCE.Convolution
    // CHECK:           VPU.NCE.Convolution
    // CHECK:           VPU.NCE.Eltwise
    // CHECK:           VPU.Yield

    // CHECK:       return [[VF]] : tensor<1x32x135x240xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NotChangeTilingStrategyForUnevenUnrolling
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x48x771x771xf16, {order = #NHWC}>)
func.func @NotChangeTilingStrategyForUnevenUnrolling(%input: tensor<1x48x771x771xf16, {order = #NHWC}>)
            -> tensor<1x32x769x769xf16, {order = #NHWC}> {
    %weightsData = const.Declare tensor<32x48x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x48x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
    %weightsSM = const.Declare tensor<32x1x1x512xi1> = dense<1.0> : tensor<32x48x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]

    %filter = VPU.GroupSparseTensor(%weightsData, %weightsSM) {sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<32xi64>, alignment = 16 : i64>, is_weights}
            -> !VPU.SparseTensor<data=tensor<32x48x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<32x1x1x512xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<32xi64>, alignment = 16 : i64>>

    %weightsTable = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %conv = VPU.NCE.Convolution(%input, %filter, %weightsTable) rawFilterShape [32, 48, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1],
        tilingStrategy = [1, 1, 55, 1]
    } : tensor<1x48x771x771xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<32x48x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<32x1x1x512xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<32xi64>, alignment = 16 : i64>>, tensor<32x1x1x4xsi32> -> tensor<1x32x769x769xf16, {order = #NHWC}>

    return %conv : tensor<1x32x769x769xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_DATA:%.+]] = const.Declare tensor<32x48x3x3xf16, {order = #NHWC}>
    // CHECK-DAG:   [[WEIGHTS_SM:%.+]] = const.Declare tensor<32x1x1x512xi1>
    // CHECK-DAG:   [[FILTER:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_DATA]], [[WEIGHTS_SM]])
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[ARG_0]], [[FILTER]], [[WEIGHTS_TABLE]]) rawFilterShape [32, 48, 3, 3] {
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>,
    // CHECK-SAME:          resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 1, 55, 1]}
    // CHECK-SAME:      -> tensor<1x32x769x769xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x769x769xf16, {order = #NHWC}>
}
