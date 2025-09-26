//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitDepthConvWithBigC
func.func @SplitDepthConvWithBigC(%arg0: tensor<1x5120x64x4xf16, {order = #NHWC}>) -> tensor<1x5120x64x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<5120x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<5120x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<5120x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [5120, 1, 1, 1], strides = [1, 1]
        } -> tensor<1x5120x64x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x64x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<5120x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<5120x1x1x4xsi32, {order = #NHWC}>
    // CHECK: [[DWConv:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[CST]], [[CST0]])
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:               rawFilterShape = [5120, 1, 1, 1], strides = [1, 1],
    // CHECK-SAME:               tilingStrategy = [1, 4, 1, 1]} -> tensor<1x5120x64x4xf16, {order = #NHWC}>
    // CHECK:  return [[DWConv]] : tensor<1x5120x64x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoSplitDepthConvOverCWithSOK
func.func @NoSplitDepthConvOverCWithSOK(%arg0: tensor<1x160x3840x4xf16, {order = #NHWC}>) -> tensor<1x160x3840x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<160x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<160x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<160x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [160, 1, 1, 1], strides = [1, 1]
        } -> tensor<1x160x3840x4xf16, {order = #NHWC}>

    return %0 : tensor<1x160x3840x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<160x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NHWC}>
    // CHECK: [[DWConv:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[CST]], [[CST0]])
    // CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:               rawFilterShape = [160, 1, 1, 1], strides = [1, 1],
    // CHECK-SAME:               tilingStrategy = [1, 1, 5, 1]} -> tensor<1x160x3840x4xf16, {order = #NHWC}>
    // CHECK:  return [[DWConv]] : tensor<1x160x3840x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolWithBigC
func.func @SplitNCEMaxPoolWithBigC(%arg0: tensor<1x5120x32x4xf16, {order = #NHWC}>) -> tensor<1x5120x32x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x5120x32x4xf16, {order = #NHWC}>

    return %0 : tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool(%arg0) {
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      } -> tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x5120x32x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolWithBigC
func.func @SplitNCEAveragePoolWithBigC(%arg0: tensor<1x5120x32x4xf16, {order = #NHWC}>) -> tensor<1x5120x32x4xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x5120x32x4xf16, {order = #NHWC}>
    return %0 : tensor<1x5120x32x4xf16, {order = #NHWC}>

    // CHECK:  [[AVGPOOL:%.+]] = VPU.NCE.AveragePool(%arg0) {
    // CHECK-SAME:   kernel_size = [1, 1],
    // CHECK-SAME:   pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:   strides = [1, 1],
    // CHECK-SAME:   tilingStrategy = [1, 2, 1, 1]} -> tensor<1x5120x32x4xf16, {order = #NHWC}>
    // CHECK:  return [[AVGPOOL]] : tensor<1x5120x32x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitSparseDepthConvWithBigC
func.func @SplitSparseDepthConvWithBigC(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [4080, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[WT:%.*]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[INPUT]], [[WT]]) {
    // CHECK:            tilingStrategy = [1, 19, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigC
func.func @SplitSparseNCEMaxPoolWithBigC(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              tilingStrategy = [1, 5, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitSparseDepthConvWithBigCWithSOK
func.func @SplitSparseDepthConvWithBigCWithSOK(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %wt) {
            ppe = #VPU.PPEStub<>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [4080, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[WT:%.*]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[INPUT]], [[WT]]) {
    // CHECK:            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:            tilingStrategy = [1, 11, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
}

// -----

// CHECK-LABEL: @TileGatherDMA
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: tensor<880x960xf16>
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: tensor<1x880xsi32>
func.func @TileGatherDMA(%arg0: tensor<880x960xf16>, %arg1: tensor<1x880xsi32>) -> tensor<1x880x960xf16> {
    %0 = VPU.Reshape(%arg1) {shape_value = [880, 1]} : tensor<1x880xsi32> -> tensor<880x1xsi32>
    %1 = VPU.Convert(%0) {dstElemType = i64} : tensor<880x1xsi32> -> tensor<880x1xi64>
    %2 = VPU.GatherDMA(%arg0, %1) {
                axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<880x960xf16>, tensor<880x1xi64> -> tensor<880x960xf16>
    %3 = VPU.Reshape(%2) {shape_value = [1, 880, 960]} : tensor<880x960xf16> -> tensor<1x880x960xf16>

    return %3 : tensor<1x880x960xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = VPU.Reshape([[INPUT_1]]) {shape_value = [880, 1]} : tensor<1x880xsi32> -> tensor<880x1xsi32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[RESHAPE_IN]]) {dstElemType = i64} : tensor<880x1xsi32> -> tensor<880x1xi64>
    // CHECK:       [[GATHER_DMA:%.+]] = VPU.GatherDMA([[INPUT_0]], [[CONVERT]]) {
    // CHECK-SAME:          axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2]} : tensor<880x960xf16>, tensor<880x1xi64> -> tensor<880x960xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [1, 880, 960]} : tensor<880x960xf16> -> tensor<1x880x960xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x880x960xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TileGatherDMA4D
// CHECK-SAME: [[INPUT_0:%arg[0-9]]]: tensor<1x30522x2100x1xf16>
// CHECK-SAME: [[INPUT_1:%arg[0-9]]]: tensor<512xsi32>
func.func @TileGatherDMA4D(%arg0: tensor<1x30522x2100x1xf16>, %arg1: tensor<512xsi32>) -> tensor<1x512x2100x1xf16> {
    %0 = VPU.Reshape(%arg1) {shape_value = [1, 512, 1, 1]} : tensor<512xsi32> -> tensor<1x512x1x1xsi32>
    %1 = VPU.Convert(%0) {dstElemType = i64} : tensor<1x512x1x1xsi32> -> tensor<1x512x1x1xi64>
    %2 = VPU.GatherDMA(%arg0, %1) {
                axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x30522x2100x1xf16>, tensor<1x512x1x1xi64> -> tensor<1x512x2100x1xf16>
    %3 = VPU.Reshape(%2) {shape_value = [1, 512, 2100, 1]} : tensor<1x512x2100x1xf16> -> tensor<1x512x2100x1xf16>

    return %3 : tensor<1x512x2100x1xf16>

    // CHECK:       [[RESHAPE_IN:%.+]] = VPU.Reshape([[INPUT_1]]) {shape_value = [1, 512, 1, 1]} : tensor<512xsi32> -> tensor<1x512x1x1xsi32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[RESHAPE_IN]]) {dstElemType = i64} : tensor<1x512x1x1xsi32> -> tensor<1x512x1x1xi64>
    // CHECK:       [[GATHER_DMA:%.+]] = VPU.GatherDMA([[INPUT_0]], [[CONVERT]]) {
    // CHECK-SAME:          axis_value = 1 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x30522x2100x1xf16>, tensor<1x512x1x1xi64> -> tensor<1x512x2100x1xf16>
    // CHECK:       [[RESHAPE_OUT:%.+]] = VPU.Reshape([[GATHER_DMA]]) {shape_value = [1, 512, 2100, 1]} : tensor<1x512x2100x1xf16> -> tensor<1x512x2100x1xf16>

    // CHECK:       return [[RESHAPE_OUT]] : tensor<1x512x2100x1xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigCWithSOK
func.func @SplitSparseNCEMaxPoolWithBigCWithSOK(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:              tilingStrategy = [1, 5, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4080x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseType = !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
!SparseType1 = !VPU.SparseTensor<data=tensor<1x4064x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4064x16x16xi1, {order = #NHWC}>>


// CHECK-LABEL: @SplitOutputSparseForConvSOKFollowedByConcat
func.func @SplitOutputSparseForConvSOKFollowedByConcat(%arg0: tensor<1x2032x16x16xf16, {order = #NHWC}>) -> tensor<1x4064x16x16xf16, {order = #NHWC}> {
    %s0 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
    %wt0 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    %maxpool0 = VPU.NCE.MaxPool(%s0, %wt0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType

    %s1 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
    %wt1 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    %maxpool1 = VPU.NCE.MaxPool(%s1, %wt1) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType


    %concat = VPU.Concat(%maxpool0, %maxpool1) {static_offsets = [[0, 0, 0, 0], [0, 2032, 0, 0]]} : !SparseType, !SparseType -> !SparseType1

    %wt2 = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
    %maxpool2 = VPU.NCE.MaxPool(%concat, %wt2) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType1

    %result = VPU.Desparsify(%maxpool2) : !SparseType1 -> tensor<1x4064x16x16xf16, {order = #NHWC}>
    return %result : tensor<1x4064x16x16xf16, {order = #NHWC}>

    // CHECK: [[ToSparsity_0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}>
    // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    // CHECK: [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[ToSparsity_0]], [[WT_0]] )
    // CHECK:              tilingStrategy = [1, 3, 1, 1]

    // CHECK: [[ToSparsity_1:%.+]] = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}>
    // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT_1:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    // CHECK: [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[ToSparsity_1]], [[WT_1]] )
    // CHECK-SAME:              tilingStrategy = [1, 3, 1, 1]

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]])
    // CHECK-DAG: [[WT_2:%.+]] = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
    // CHECK: [[MAXPOOL_2:%.+]] = VPU.NCE.MaxPool([[CONCAT]], [[WT_2]] )
    // CHECK-SAME:              tilingStrategy = [1, 5, 1, 1]
    // CHECK: [[RESULT:%.+]] = VPU.Desparsify([[MAXPOOL_2]])

    // CHECK: return [[RESULT]]
}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0043085547638874429:24>

// CHECK-LABEL: @DontTileD2SDMA
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x64x128x128x!qElemType, {order = #NHWC}>
func.func @DontTileD2SDMA(%arg0: tensor<1x64x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
    %avgpool = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]}
            -> tensor<1x64x128x128x!qElemType, {order = #NHWC}>
    %d2s = VPU.DepthToSpace(%avgpool) {
        block_size = 2 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x64x128x128x!qElemType, {order = #NHWC}>
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    %eltwise = VPU.NCE.Eltwise(%d2s, %d2s) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
            -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
    return %eltwise : tensor<1x16x256x256x!qElemType, {order = #NHWC}>

    // CHECK:       [[AVGPOOL:%.+]] = VPU.NCE.AveragePool([[INPUT]])
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[AVGPOOL]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 1]
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[D2S]], [[D2S]])
    // CHECK:       return [[ELTWISE]]
}

// -----

// CHECK-LABEL: @MVNSOKAndKTile
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x32x61440x1xf16>
func.func @MVNSOKAndKTile(%arg0: tensor<1x32x61440x1xf16>) -> tensor<1x32x61440x1xf16> {
    %mvn = VPU.MVN(%arg0) {
        across_channels = false, eps = 9.9999997473787516E-6 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        normalize_variance = true
    } : tensor<1x32x61440x1xf16> -> tensor<1x32x61440x1xf16>

    return %mvn : tensor<1x32x61440x1xf16>

    // CHECK:       VPU.MVN([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndGTile
func.func @NCEMatMulSOGAndGTile(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %cst_0 = const.Declare tensor<512x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst_0) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 64, 32, 1, 1], strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 1, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndHTile
func.func @NCEMatMulSOGAndHTile(%arg0: tensor<6x1x512x512xf16>, %arg1: tensor<6x1x512x512xf16>) -> tensor<6x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<6x512x1x1x4xsi32> = dense<10> : tensor<6x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<6x512x512x1x1xf16> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<6x512x512x1x1xf16> -> tensor<6x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [6, 1, 512, 128, 4]} : tensor<6x1x512x512x1xf16, {order = #GNHWC}> -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [6, 512, 512, 1, 1], strides = [1, 1]} -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [6, 1, 512, 512, 1]} : tensor<6x1x512x128x4xf16, {order = #GNHWC}> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<6x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 2, 1]
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndGHTile
func.func @NCEMatMulSOGAndGHTile(%arg0: tensor<12x1x512x512xf16>, %arg1: tensor<12x1x512x512xf16>) -> tensor<12x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<12x512x1x1x4xsi32> = dense<10> : tensor<12x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 12, 512, 512]} : tensor<12x1x512x512xf16> -> tensor<1x12x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 12, 512, 512]} : tensor<12x1x512x512xf16> -> tensor<1x12x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 512, 512, 1, 1]} : tensor<1x12x512x512xf16> -> tensor<12x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<12x512x512x1x1xf16> -> tensor<12x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [12, 512, 512, 1, 1]} : tensor<1x12x512x512xf16> -> tensor<12x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<12x512x512x1x1xf16> -> tensor<12x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [12, 1, 512, 128, 4]} : tensor<12x1x512x512x1xf16, {order = #GNHWC}> -> tensor<12x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [12, 512, 512, 1, 1], strides = [1, 1]} -> tensor<12x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [12, 1, 512, 512, 1]} : tensor<12x1x512x128x4xf16, {order = #GNHWC}> -> tensor<12x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<12x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 7, 1]
}

// -----

// CHECK-LABEL:   func.func @NotSplitGridSample
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1x1920x1088xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1920x1088x2xf16>
func.func @NotSplitGridSample(%arg0: tensor<1x1x1920x1088xf16>, %arg1: tensor<1x1920x1088x2xf16>) -> tensor<1x1x1920x1088xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x1920x1088xf16>, tensor<1x1920x1088x2xf16> -> tensor<1x1x1920x1088xf16>
    return %0 : tensor<1x1x1920x1088xf16>

    // CHECK:       [[GRID_SAMPLE:%.+]] = VPU.GridSample([[INPUT1]], [[INPUT2]]) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>}
    // CHECK-SAME:     : tensor<1x1x1920x1088xf16>, tensor<1x1920x1088x2xf16> -> tensor<1x1x1920x1088xf16>
    // CHECK:       return [[GRID_SAMPLE]] : tensor<1x1x1920x1088xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGridSample
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x4x960x544xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x960x544x2xf16>
func.func @SplitGridSample(%arg0: tensor<1x4x960x544xf16>, %arg1: tensor<1x960x544x2xf16>) -> tensor<1x4x960x544xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x4x960x544xf16>, tensor<1x960x544x2xf16> -> tensor<1x4x960x544xf16>
    return %0 : tensor<1x4x960x544xf16>

    // CHECK:       [[GRID_SAMPLE:%.+]] = VPU.GridSample([[INPUT1]], [[INPUT2]]) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>,
    // CHECK-SAME:          tilingStrategy = [1, 4, 8, 1]}
    // CHECK-SAME:     : tensor<1x4x960x544xf16>, tensor<1x960x544x2xf16> -> tensor<1x4x960x544xf16>

    // CHECK:       return [[GRID_SAMPLE]] : tensor<1x4x960x544xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitWithoutOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x16x32x56x16xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x14580x2xsi32>
func.func @GatherNDSplitWithoutOriginalShape(%arg0: tensor<1x16x32x56x16xf16>, %arg1: tensor<1x16x14580x2xsi32>) -> tensor<1x16x14580x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 2 : i64
                    } : tensor<1x16x32x56x16xf16>, tensor<1x16x14580x2xsi32> -> tensor<1x16x14580x16xf16>

    return %0 : tensor<1x16x14580x16xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.GatherND([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:               batch_dims = 2 : i64,
    // CHECK-SAME:               tilingStrategy = [1, 8, 1, 1]}
    // CHECK-SAME:           : tensor<1x16x32x56x16xf16>, tensor<1x16x14580x2xsi32> -> tensor<1x16x14580x16xf16>

    // CHECK: return [[GATHER]] : tensor<1x16x14580x16xf16>
}

// -----

// CHECK-LABEL: func.func @GatherND4DSplitWithOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x16x1792x16xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x16x14580x2xsi32>
func.func @GatherND4DSplitWithOriginalShape(%arg0: tensor<1x16x1792x16xf16>, %arg1: tensor<1x16x14580x2xsi32>) -> tensor<1x16x14580x16xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 2 : i64, original_shape = [1, 16, 32, 56, 16]
                    } : tensor<1x16x1792x16xf16>, tensor<1x16x14580x2xsi32> -> tensor<1x16x14580x16xf16>

    return %0 : tensor<1x16x14580x16xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.GatherND([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:               batch_dims = 2 : i64,
    // CHECK-SAME:               original_shape = [1, 16, 32, 56, 16],
    // CHECK-SAME:               tilingStrategy = [1, 8, 1, 1]}
    // CHECK-SAME:           : tensor<1x16x1792x16xf16>, tensor<1x16x14580x2xsi32> -> tensor<1x16x14580x16xf16>

    // CHECK: return [[GATHER]] : tensor<1x16x14580x16xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitAtIndicesWithoutOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x9x16x8xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x104580x2xsi32>
func.func @GatherNDSplitAtIndicesWithoutOriginalShape(%arg0: tensor<1x9x16x8xf16>, %arg1: tensor<1x104580x2xsi32>) -> tensor<1x104580x8xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 1 : i64
                    } : tensor<1x9x16x8xf16>, tensor<1x104580x2xsi32> -> tensor<1x104580x8xf16>

    return %0 : tensor<1x104580x8xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.GatherND([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:               batch_dims = 1 : i64,
    // CHECK-SAME:               tilingStrategy = [1, 2, 1]}
    // CHECK-SAME:           : tensor<1x9x16x8xf16>, tensor<1x104580x2xsi32> -> tensor<1x104580x8xf16>

    // CHECK: return [[GATHER]] : tensor<1x104580x8xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitAtIndicesWithOriginalShape
// CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x1x144x8xf16>
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x1x104580x2xsi32>
func.func @GatherNDSplitAtIndicesWithOriginalShape(%arg0: tensor<1x1x144x8xf16>, %arg1: tensor<1x1x104580x2xsi32>) -> tensor<1x1x104580x8xf16> {
    %0 = VPU.GatherND(%arg0, %arg1) {batch_dims = 2 : i64, original_shape = [1, 1, 9, 16, 8]
                    } : tensor<1x1x144x8xf16>, tensor<1x1x104580x2xsi32> -> tensor<1x1x104580x8xf16>

    return %0 : tensor<1x1x104580x8xf16>

    // CHECK:       [[GATHER:%.+]] = VPU.GatherND([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:               batch_dims = 2 : i64,
    // CHECK-SAME:               tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:           : tensor<1x1x144x8xf16>, tensor<1x1x104580x2xsi32> -> tensor<1x1x104580x8xf16>

    // CHECK: return [[GATHER]] : tensor<1x1x104580x8xf16>
}

// -----

// CHECK-LABEL:   @MultiplyNotAlign
// CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x512x48x336xf16>,
// CHECK-SAME:     [[INPUT1:%.+]]: tensor<1x512x48x336xf16>)
func.func @MultiplyNotAlign(%arg0: tensor<1x512x48x336xf16>, %arg1: tensor<1x512x48x336xf16>) -> tensor<1x512x48x336xf16> {
    %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<1x512x48x336xf16>, tensor<1x512x48x336xf16> -> tensor<1x512x48x336xf16>

    return %0 : tensor<1x512x48x336xf16>


    // CHECK: tilingStrategy = [1, 35, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SEPDWConvCTile
func.func @SEPDWConvCTile(%arg0: tensor<1x768x32x32xf16, {order = #NHWC}>) -> tensor<1x768x16x16xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<1x768x16x16xi1, {order = #NHWC}> = dense<1> : tensor<1x768x16x16xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
  %cst_0 = const.Declare tensor<768x1x1x4xsi32> = dense<1> : tensor<768x1x1x4xsi32>
  %cst_1 = const.Declare tensor<768x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<768x1x1x3x3xf16>, [#const.Reshape<[768, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 7, 0, 0]>, #const.Reorder<#NHWC>]
  %storage_elem_table = VPU.StorageElementTable {
    dataElemType = f16, dataShape = [1, 768, 32, 32],
    seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0], dataSizes = [1, 768, 32, 32]>,
    seDepth = 12 : i64, seSize = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]}
            -> tensor<1x12x16x16xi32, {order = #NHWC}>
  %group_sparse_tensor = VPU.GroupSparseTensor(%arg0, %cst, %storage_elem_table) {seAttr = #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3],
        dataOffset = [0, 0, 0, 0], dataSizes = [1, 768, 32, 32]>}
            -> !VPU.SparseTensor<data=tensor<1x768x32x32xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x768x16x16xi1, {order = #NHWC}>,
            storage_element_table=tensor<1x12x16x16xi32, {order = #NHWC}>,
            #VPU.SEDilatedConv<dilation = [2, 2], kernelStride = [1, 1], kernelSize = [3, 3], dataOffset = [0, 0, 0, 0],
            dataSizes = [1, 768, 32, 32]>>
  %4 = VPU.NCE.DepthConvolution(%group_sparse_tensor, %cst_1, %cst_0) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>,
        clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [768, 1, 3, 3], strides = [1, 1]}
            -> tensor<1x768x16x16xf16, {order = #NHWC}>
  return %4 : tensor<1x768x16x16xf16, {order = #NHWC}>

  // Tile the SEP Dilated DWConv to the supported workload channel sizes
  // CHECK:         VPU.NCE.DepthConvolution
  // CHECK-SAME:    tilingStrategy = [1, 2, 1, 1]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

// CHECK-LABEL:   @SplitI4QuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
func.func @SplitI4QuantNCEConvOverOC(%arg0: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x6320x256x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<6320x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<6320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
    } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}>, tensor<6320x1x1x4xsi32, {order = #NCHW}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    return %0 : tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<6320x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<6320x1x1x4xsi32>

    // CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          rawFilterShape = [6320, 128, 1, 1],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 12, 1, 1]}
    // CHECK-SAME:          -> tensor<1x6320x256x4xf16, {order = #NHWC}>

    // CHECK:           return [[CONV]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @NCEMatMulSOGAndCTile
// CHECK-SAME:      [[INPUT:%.+]]: tensor<16x1x128x1x4xf16, {order = #GNHWC}>
func.func @NCEMatMulSOGAndCTile(%arg0: tensor<16x1x128x1x4xf16, {order = #GNHWC}>) -> tensor<16x1x4224x1x4xf16, {order = #GNHWC}> {
    %weights = const.Declare tensor<16x4224x128x1x1xf16, {order = #GNHWC}> = dense<1.0> : tensor<16x4224x128x1x1xf16, {order = #GNHWC}>
    %weights_table = const.Declare tensor<16x4224x1x1x4xsi32> = dense<0> : tensor<16x4224x1x1x4xsi32>

    %grouped_matmul = VPU.NCE.MatMul(%arg0, %weights, %weights_table) {
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        rawFilterShape = [16, 4224, 128, 1, 1],
        strides = [1, 1]
    } -> tensor<16x1x4224x1x4xf16, {order = #GNHWC}>

    return %grouped_matmul : tensor<16x1x4224x1x4xf16, {order = #GNHWC}>

    // CHECK:         VPU.NCE.MatMul
    // CHECK-SAME:    tilingStrategy = [1, 1, 3, 1, 1]
}
