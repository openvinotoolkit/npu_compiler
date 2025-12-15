//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --vertical-fusion-tiling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.003883181594488189:127,0.0031930517962598425:127,0.0036140501968503938:127,0.0036563422736220473:127,0.0035063976377952754:127,0.0039908341535433069:127,0.0036659541092519685:127,0.003196896530511811:127,0.0035217765748031494:127,0.0032622570127952754:127,0.0038408895177165355:127,0.0035256213090551179:127,0.0038332000492125986:127,0.003371831938976378:127,0.0035813699557086616:127,0.0037024790846456692:127,0.0038197434793307088:127,0.0036121278297244095:127,0.0033449187992125986:127,0.0031161571112204725:127,0.0036505751722440945:127,0.0034890963336614172:127,0.0038735697588582678:127,0.0033756766732283465:127,0.0030584860974409451:127,0.0037178580216535432:127,0.003456416092519685:127,0.0033256951279527561:127,0.0033487635334645671:127,0.0041484682578740153:127,0.0041215551181102358:127,0.0034910187007874014:127}>

func.func @VfTilingWithEltwise(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %weights_1: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %weights_2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>  {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %weights_1 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %weights_2 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }

    return %0 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK: [[SLICEARG0TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 130, 256]
    // CHECK: [[CONV0TILE0:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE0]], %arg1)
    // CHECK-SAME: {
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:   rawFilterShape = [32, 16, 3, 3], strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x32x129x256x!qElemType, {order = #NHWC}>
    // CHECK: [[CONV1TILE0:%.+]] = VPU.NCE.Convolution([[CONV0TILE0]], %arg2)
    // CHECK-SAME: {
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:  rawFilterShape = [32, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:  -> tensor<1x32x128x256x!qElemType, {order = #NHWC}>
    // CHECK: [[SLICETILE0:%.+]] = VPU.Slice [[CONV0TILE0]] [0, 0, 0, 0] [1, 32, 128, 256]
    // CHECK: [[ELTWISETILE0:%.+]] = VPU.NCE.Eltwise([[SLICETILE0]], [[CONV1TILE0]])
    // CHECK: [[SLICEARG0TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 126, 0] [1, 16, 130, 256]
    // CHECK: [[CONV0TILE1:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE1]], %arg1)
    // CHECK-SAME: {
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:   rawFilterShape = [32, 16, 3, 3],
   // CHECK-SAME:    strides = [1, 1]}
   // CHECK-SAME:    -> tensor<1x32x129x256x!qElemType, {order = #NHWC}>
    // CHECK: [[CONV1TILE1:%.+]] = VPU.NCE.Convolution([[CONV0TILE1]], %arg2)
    // CHECK-SAME: {
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:   rawFilterShape = [32, 32, 3, 3],
    // CHECK-SAME:   strides = [1, 1]}
    // CHECK-SAME:   -> tensor<1x32x128x256x!qElemType, {order = #NHWC}>
    // CHECK: [[SLICETILE1:%.+]] = VPU.Slice [[CONV0TILE1]] [0, 0, 1, 0] [1, 32, 128, 256]
    // CHECK: [[ELTWISETILE1:%.+]] = VPU.NCE.Eltwise([[SLICETILE1]], [[CONV1TILE1]])
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISETILE0]], [[ELTWISETILE1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 128, 0]]} : tensor<1x32x128x256x!qElemType, {order = #NHWC}>, tensor<1x32x128x256x!qElemType, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
    // CHECK: return [[CONCAT]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VfTilingWithEltwiseAdjustOffset
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
// CHECK-SAME:      [[W1:%.+]]: tensor<32x16x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [[W2:%.+]]: tensor<32x32x7x7xf16, {order = #NHWC}>
func.func @VfTilingWithEltwiseAdjustOffset(
            %arg0: tensor<1x16x128x128xf16, {order = #NHWC}>, %weights_1: tensor<32x16x3x3xf16, {order = #NHWC}>,
            %weights_2: tensor<32x32x7x7xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>  {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x128x128xf16, {order = #NHWC}>, %weights_1 as %arg2: tensor<32x16x3x3xf16, {order = #NHWC}>,
                            %weights_2 as %arg4: tensor<32x32x7x7xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x128x128xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 7, 7], strides = [1, 1]} : tensor<1x32x128x128xf16, {order = #NHWC}>, tensor<32x32x7x7xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK:       [[HEAD_IN_SLICE_1:%.+]] = VPU.Slice {{.*}} [0, 0, 0, 0] [1, 16, 68, 128] : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x68x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV1_1:%.+]] = VPU.NCE.Convolution([[HEAD_IN_SLICE_1]]
    // CHECK:       [[CONV2_1:%.+]] = VPU.NCE.Convolution([[CONV1_1]]
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[CONV1_1]] [0, 0, 0, 0] [1, 32, 64, 128]
    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[SLICE_1]], [[CONV2_1]])

    // CHECK:       [[HEAD_IN_SLICE_2:%.+]] = VPU.Slice {{.*}} [0, 0, 60, 0] [1, 16, 68, 128] : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x68x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV1_2:%.+]] = VPU.NCE.Convolution([[HEAD_IN_SLICE_2]]
    // CHECK:       [[CONV2_2:%.+]] = VPU.NCE.Convolution([[CONV1_2]]
    // CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[CONV1_2]] [0, 0, 3, 0] [1, 32, 64, 128]
    // CHECK:       [[ELTWISE_2:%.+]] = VPU.NCE.Eltwise([[SLICE_2]], [[CONV2_2]])

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_1]], [[ELTWISE_2]])
    // CHECK:       return [[CONCAT]] : tensor<1x32x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @TileGroupSparseTensor(%arg0: tensor<1x32x24x30xf16, {order = #NHWC}>) -> tensor<1x16x48x60xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<1x32x49x61xi1, {order = #NHWC}> = dense<1> : tensor<1x32x49x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %cst_1 = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %0 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    %1 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x24x30xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<1x32x49x61xi1, {order = #NHWC}>, %0 as %arg3: tensor<1x1x49x61xi32, {order = #NHWC}>, %cst_1 as %arg4: tensor<16x32x2x2xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x48x60xf16, {order = #NHWC}> {
      %2 = VPU.GroupSparseTensor(%arg1, %arg2, %arg3) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>} -> !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>
      %3 = VPU.NCE.Convolution(%2, %arg4) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 32, 2, 2], strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x48x60xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %1 : tensor<1x16x48x60xf16, {order = #NHWC}>

    // CHECK: [[SET:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    // CHECK: [[SLICE_ARG_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 12, 30]
    // CHECK: [[SLICE_CST_0:%.+]] = VPU.Slice %cst [0, 0, 0, 0] [1, 32, 25, 61]
    // CHECK: [[SLICE_SET_0:%.+]] = VPU.Slice [[SET]] [0, 0, 0, 0] [1, 1, 25, 61]
    // CHECK: [[GST0:%.+]] = VPU.GroupSparseTensor([[SLICE_ARG_0]], [[SLICE_CST_0]], [[SLICE_SET_0]])
    // CHECK-SAME: {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2], offsets = [0, 0, 0, 0], sizes = [1, 32, 25, 61]>
    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[GST0]], %cst_0)
    // CHECK-SAME: tensor<1x16x24x60xf16, {order = #NHWC}>
    // CHECK: [[SLICE_ARG_1:%.+]] = VPU.Slice %arg0 [0, 0, 11, 0] [1, 32, 13, 30]
    // CHECK: [[SLICE_CST_1:%.+]] = VPU.Slice %cst [0, 0, 24, 0] [1, 32, 25, 61]
    // CHECK: [[SLICE_SET_1:%.+]] = VPU.Slice [[SET]] [0, 0, 24, 0] [1, 1, 25, 61]
    // CHECK: [[GST1:%.+]] = VPU.GroupSparseTensor([[SLICE_ARG_1]], [[SLICE_CST_1]], [[SLICE_SET_1]])
    // CHECK-SAME: {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2], offsets = [0, 0, 2, 0], sizes = [1, 32, 25, 61]>
    // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[GST1]], %cst_0)
    // CHECK-SAME: tensor<1x16x24x60xf16, {order = #NHWC}>
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 24, 0]]}
    // CHECK: return [[CONCAT]] : tensor<1x16x48x60xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileGroupSparseTensorWithOffset
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x24x30xf16, {order = #NHWC}>
func.func @TileGroupSparseTensorWithOffset(%input: tensor<1x32x24x30xf16, {order = #NHWC}>) -> tensor<1x16x48x58xf16, {order = #NHWC}> {
    %cst_weights = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %cst_sparsity_map = const.Declare tensor<1x32x49x59xi1, {order = #NHWC}> = dense<1> : tensor<1x32x49x59xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %se_table = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1], offsets = [0, 0, 0, 2]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x59xi32, {order = #NHWC}>
    %vf = VPU.VerticalFusion (%input as %vf_input: tensor<1x32x24x30xf16, {order = #NHWC}>, %cst_sparsity_map as %sparsity_map: tensor<1x32x49x59xi1, {order = #NHWC}>, %se_table as %vf_se_table: tensor<1x1x49x59xi32, {order = #NHWC}>, %cst_weights as %weights: tensor<16x32x2x2xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x48x58xf16, {order = #NHWC}> {
      %sparse_tensor = VPU.GroupSparseTensor(%vf_input, %sparsity_map, %vf_se_table) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1], offsets = [0, 0, 0, 2]>} -> !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x59xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x59xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1], offsets = [0, 0, 0, 2]>>
      %conv = VPU.NCE.Convolution(%sparse_tensor, %weights) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 32, 2, 2], strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x59xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x59xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1], offsets = [0, 0, 0, 2]>>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x48x58xf16, {order = #NHWC}>
      VPU.Yield %conv
    }
    return %vf : tensor<1x16x48x58xf16, {order = #NHWC}>

    // CHECK-DAG: VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 12, 30]
    // CHECK-DAG: VPU.Slice [[INPUT]] [0, 0, 11, 0] [1, 32, 13, 30]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0094078685723099058:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0047039342861549529:128>

// CHECK-LABEL: @VfTilingWithQuantizeCast
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x48x48x!qElemType, {order = #NHWC}>
func.func @VfTilingWithQuantizeCast(%arg0: tensor<1x32x48x48x!qElemType, {order = #NHWC}>) -> tensor<1x32x48x48xf16, {order = #NHWC}> {
   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x48x48x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x48x48xf16, {order = #NHWC}> {
      %1 = VPU.QuantizeCast(%arg1) {dstElemType = !qElemType1} : tensor<1x32x48x48x!qElemType, {order = #NHWC}> -> tensor<1x32x48x48x!qElemType1, {order = #NHWC}>
      %2 = VPU.NCE.Eltwise(%1, %1)
         {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x48x48xf16, {order = #NHWC}>
      VPU.Yield %2
   }
   return %0 : tensor<1x32x48x48xf16, {order = #NHWC}>

   // CHECK:         [[SLICE_ARG_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 24, 48]
   // CHECK:         [[QUANTIZE_CAST_0:%.+]] = VPU.QuantizeCast([[SLICE_ARG_0]]) {dstElemType = !qElemType1}
   // CHECK-SAME:        tensor<1x32x24x48x!qElemType, {order = #NHWC}> -> tensor<1x32x24x48x!qElemType1, {order = #NHWC}>
   // CHECK:         [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[QUANTIZE_CAST_0]], [[QUANTIZE_CAST_0]]) {op_type = #VPU.eltwise_type<ADD>
   // CHECK-SAME:        tensor<1x32x24x48xf16, {order = #NHWC}>
   // CHECK:         [[SLICE_ARG_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 24, 0] [1, 32, 24, 48]
   // CHECK:         [[QUANTIZE_CAST_1:%.+]] = VPU.QuantizeCast([[SLICE_ARG_1]]) {dstElemType = !qElemType1}
   // CHECK-SAME:        tensor<1x32x24x48x!qElemType, {order = #NHWC}> -> tensor<1x32x24x48x!qElemType1, {order = #NHWC}>
   // CHECK:         [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[QUANTIZE_CAST_1]], [[QUANTIZE_CAST_1]]) {op_type = #VPU.eltwise_type<ADD>
   // CHECK-SAME:        tensor<1x32x24x48xf16, {order = #NHWC}>
   // CHECK:         [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
   // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 24, 0]]}
   // CHECK-SAME:        tensor<1x32x24x48xf16, {order = #NHWC}>, tensor<1x32x24x48xf16, {order = #NHWC}> -> tensor<1x32x48x48xf16, {order = #NHWC}>
   // CHECK:         return [[CONCAT]] : tensor<1x32x48x48xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FallBackToOperandDueToSliceFail
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x720x1280xf16, {order = #NHWC}>
func.func @FallBackToOperandDueToSliceFail(%arg0: tensor<1x16x720x1280xf16, {order = #NHWC}>) -> tensor<1x16x720x1280xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x720x1280xf16, {order = #NHWC}>, %cst as %arg2: tensor<16x16x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 32]} -> tensor<1x16x720x1280xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [16, 16, 3, 3], strides = [2, 2]} : tensor<1x16x720x1280xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}> -> tensor<1x16x359x639xf16, {order = #NHWC}>
      %2 = VPU.NCE.MaxPool(%1) {kernel_size = [7, 7], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, strides = [3, 3]} -> tensor<1x16x118x211xf16, {order = #NHWC}>
      %4 = VPU.Interpolate(%2) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <PYTORCH_HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 16, 118, 211], initial_output_dims_attr = [1, 16, 720, 1280], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [6.101694915254237, 6.0663507109004735], sizes_attr = [720, 1280], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x16x118x211xf16, {order = #NHWC}> -> tensor<1x16x720x1280xf16, {order = #NHWC}>
      %6 = VPU.NCE.Eltwise(%arg1, %4) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x16x720x1280xf16, {order = #NHWC}>
      VPU.Yield %6
    }
    return %0 : tensor<1x16x720x1280xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK: [[CONV_SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 720, 57]
    // CHECK: [[CONV_0:%.+]] = VPU.NCE.Convolution([[CONV_SLICE_0]], [[WEIGHTS]])
    // CHECK: [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[CONV_0]])
    // CHECK: [[INTERP_0:%.+]] = VPU.Interpolate([[MAXPOOL_0]])
    // CHECK: [[ELTWISE_SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 720, 57]
    // CHECK: [[ELTWISE_INPUT_0:%.+]] = VPU.Slice [[ELTWISE_SLICE_0]] [0, 0, 0, 0] [1, 16, 720, 40]
    // CHECK: [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT_0]], [[INTERP_0]])

    // CHECK: [[CONV_SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 72] [1, 16, 720, 63]
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_SLICE_1]], [[WEIGHTS]])
    // CHECK: [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[CONV_1]])
    // CHECK: [[INTERP_1:%.+]] = VPU.Interpolate([[MAXPOOL_1]])
    // CHECK: [[ELTWISE_SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 72] [1, 16, 720, 63]
    // CHECK: [[ELTWISE_INPUT_1:%.+]] = VPU.Slice [[ELTWISE_SLICE_1]] [0, 0, 0, 8] [1, 16, 720, 40]
    // CHECK: [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT_1]], [[INTERP_1]])

    // branch2 ~ branch 29

    // CHECK: [[CONV_SLICE_30:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1182] [1, 16, 720, 57]
    // CHECK: [[CONV_30:%.+]] = VPU.NCE.Convolution([[CONV_SLICE_30]], [[WEIGHTS]])
    // CHECK: [[MAXPOOL_30:%.+]] = VPU.NCE.MaxPool([[CONV_30]])
    // CHECK: [[INTERP_30:%.+]] = VPU.Interpolate([[MAXPOOL_30]])

    //   "invalid offsets: Input Offset 1182, shape 57 ==> offset: 1200, shape: 40"
    //   The input of ELTWISE_INPUT_30 is arg instead of ELTWISE_SLICE_30

    // CHECK: [[ELTWISE_SLICE_30:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1182] [1, 16, 720, 57]
    // CHECK: [[ELTWISE_INPUT_30:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1200] [1, 16, 720, 40]
    // CHECK: [[ELTWISE_30:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT_30]], [[INTERP_30]])

    // CHECK: [[CONV_SLICE_31:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1218] [1, 16, 720, 57]
    // CHECK: [[CONV_31:%.+]] = VPU.NCE.Convolution([[CONV_SLICE_31]], [[WEIGHTS]])
    // CHECK: [[MAXPOOL_31:%.+]] = VPU.NCE.MaxPool([[CONV_31]])
    // CHECK: [[INTERP_31:%.+]] = VPU.Interpolate([[MAXPOOL_31]])

    //   "invalid offsets: Input Offset 1218, shape 57 ==> offset: 1240, shape: 40"
    //   The input of ELTWISE_INPUT_31 is arg instead of ELTWISE_SLICE_31

    // CHECK: [[ELTWISE_SLICE_31:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1218] [1, 16, 720, 57]
    // CHECK: [[ELTWISE_INPUT_31:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 1240] [1, 16, 720, 40]
    // CHECK: [[ELTWISE_31:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT_31]], [[INTERP_31]])

    // CHECK: [[CONCAT:%.+]] = VPU.Concat
    // CHECK: return [[CONCAT]] : tensor<1x16x720x1280xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @InterpAndAvgpoolPropagateAxis
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x64x96x160xf16, {order = #NHWC}>
func.func @InterpAndAvgpoolPropagateAxis(%arg0: tensor<1x64x96x160xf16, {order = #NHWC}>) -> tensor<1x64x192x320xf16, {order = #NHWC}> {

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x96x160xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 5]} -> tensor<1x64x192x320xf16, {order = #NHWC}> {
      %1 = VPU.Interpolate(%arg1) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ALIGN_CORNERS>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], initial_input_dims_attr = [1, 64, 96, 160], initial_output_dims_attr = [1, 64, 192, 320], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320], tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} : tensor<1x64x96x160xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
      %2 = VPU.NCE.AveragePool(%1) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [41.160151324085753], fp_prelu_alpha = 41.160152435302734 : f64>, strides = [1, 1]} -> tensor<1x64x192x320xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 64, 96, 33]
    // CHECK: [[INTERP_0:%.+]] = VPU.Interpolate([[SLICE_0]])
    // CHECK: [[AVGPOOL_0:%.+]] = VPU.NCE.AveragePool([[INTERP_0]])

    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 31] [1, 64, 96, 34]
    // CHECK: [[INTERP_1:%.+]] = VPU.Interpolate([[SLICE_1]])
    // CHECK: [[AVGPOOL_1:%.+]] = VPU.NCE.AveragePool([[INTERP_1]])

    // CHECK: [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 63] [1, 64, 96, 34]
    // CHECK: [[INTERP_2:%.+]] = VPU.Interpolate([[SLICE_2]])
    // CHECK: [[AVGPOOL_2:%.+]] = VPU.NCE.AveragePool([[INTERP_2]])

    // CHECK: [[SLICE_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 95] [1, 64, 96, 34]
    // CHECK: [[INTERP_3:%.+]] = VPU.Interpolate([[SLICE_3]])
    // CHECK: [[AVGPOOL_3:%.+]] = VPU.NCE.AveragePool([[INTERP_3]])

    // CHECK: [[SLICE_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 127] [1, 64, 96, 33]
    // CHECK: [[INTERP_4:%.+]] = VPU.Interpolate([[SLICE_4]])
    // CHECK: [[AVGPOOL_4:%.+]] = VPU.NCE.AveragePool([[INTERP_4]])

    // CHECK: [[CONCAT:%.+]] = VPU.Concat
    // CHECK: return [[CONCAT]] : tensor<1x64x192x320xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VfTilingWithMultiEltwiseAdjustOffset
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
// CHECK-SAME:      [[W1:%.+]]: tensor<32x16x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [[W2:%.+]]: tensor<32x32x7x7xf16, {order = #NHWC}>
func.func @VfTilingWithMultiEltwiseAdjustOffset(
            %arg0: tensor<1x16x128x128xf16, {order = #NHWC}>, %weights_1: tensor<32x16x3x3xf16, {order = #NHWC}>,
            %weights_2: tensor<32x32x7x7xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>  {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x128x128xf16, {order = #NHWC}>, %weights_1 as %arg2: tensor<32x16x3x3xf16, {order = #NHWC}>,
                            %weights_2 as %arg4: tensor<32x32x7x7xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x128x128xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 7, 7], strides = [1, 1]} : tensor<1x32x128x128xf16, {order = #NHWC}>, tensor<32x32x7x7xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 7, 7], strides = [1, 1]} : tensor<1x32x128x128xf16, {order = #NHWC}>, tensor<32x32x7x7xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%1, %4)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x128x128xf16, {order = #NHWC}>
      VPU.Yield %5
    }
    return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK:       [[HEAD_IN_SLICE_1:%.+]] = VPU.Slice {{.*}} [0, 0, 0, 0] [1, 16, 71, 128] : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x71x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV1_1:%.+]] = VPU.NCE.Convolution([[HEAD_IN_SLICE_1]]
    // CHECK:       [[CONV2_1:%.+]] = VPU.NCE.Convolution([[CONV1_1]]
    // CHECK:       [[SLICE1_1:%.+]] = VPU.Slice [[CONV1_1]] [0, 0, 0, 0] [1, 32, 67, 128]
    // CHECK:       [[ELTWISE1_1:%.+]] = VPU.NCE.Eltwise([[SLICE1_1]], [[CONV2_1]])
    // CHECK:       [[CONV3_1:%.+]] = VPU.NCE.Convolution([[ELTWISE1_1]]
    // CHECK:       [[SLICE2_1:%.+]] = VPU.Slice [[SLICE1_1]] [0, 0, 0, 0] [1, 32, 64, 128]
    // CHECK:       [[ELTWISE2_1:%.+]] = VPU.NCE.Eltwise([[SLICE2_1]], [[CONV3_1]])

    // CHECK:       [[HEAD_IN_SLICE_2:%.+]] = VPU.Slice {{.*}} [0, 0, 57, 0] [1, 16, 71, 128] : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x71x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV1_2:%.+]] = VPU.NCE.Convolution([[HEAD_IN_SLICE_2]]
    // CHECK:       [[CONV2_2:%.+]] = VPU.NCE.Convolution([[CONV1_2]]
    // CHECK:       [[SLICE1_2:%.+]] = VPU.Slice [[CONV1_2]] [0, 0, 3, 0] [1, 32, 67, 128]
    // CHECK:       [[ELTWISE1_2:%.+]] = VPU.NCE.Eltwise([[SLICE1_2]], [[CONV2_2]])
    // CHECK:       [[CONV3_2:%.+]] = VPU.NCE.Convolution([[ELTWISE1_2]]
    // CHECK-NOT:   [[SLICE2_2:%.+]] = VPU.Slice [[SLICE1_2]] [0, 0, 6, 0] [1, 32, 64, 128]
    // CHECK:       [[SLICE2_2:%.+]] = VPU.Slice [[SLICE1_2]] [0, 0, 3, 0] [1, 32, 64, 128]
    // CHECK:       [[ELTWISE2_2:%.+]] = VPU.NCE.Eltwise([[SLICE2_2]], [[CONV3_2]])
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE2_1]], [[ELTWISE2_2]])
    // CHECK:       return [[CONCAT]] : tensor<1x32x128x128xf16, {order = #NHWC}>
}
