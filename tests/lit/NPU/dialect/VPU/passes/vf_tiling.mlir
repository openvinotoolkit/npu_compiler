//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --vertical-fusion-tiling %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.003883181594488189:127,0.0031930517962598425:127,0.0036140501968503938:127,0.0036563422736220473:127,0.0035063976377952754:127,0.0039908341535433069:127,0.0036659541092519685:127,0.003196896530511811:127,0.0035217765748031494:127,0.0032622570127952754:127,0.0038408895177165355:127,0.0035256213090551179:127,0.0038332000492125986:127,0.003371831938976378:127,0.0035813699557086616:127,0.0037024790846456692:127,0.0038197434793307088:127,0.0036121278297244095:127,0.0033449187992125986:127,0.0031161571112204725:127,0.0036505751722440945:127,0.0034890963336614172:127,0.0038735697588582678:127,0.0033756766732283465:127,0.0030584860974409451:127,0.0037178580216535432:127,0.003456416092519685:127,0.0033256951279527561:127,0.0033487635334645671:127,0.0041484682578740153:127,0.0041215551181102358:127,0.0034910187007874014:127}>

func.func @VfTilingWithEltwise(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %wt: tensor<32x1x1x4xsi32>, %weights_1: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %weights_2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>  {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %weights_1 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %wt as %arg3: tensor<32x1x1x4xsi32>, %weights_2 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }

    return %0 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK: [[SLICEARG0TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 16, 130, 256]
    // CHECK: [[CONV0TILE0:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE0]], %arg2, %arg1)
    // CHECK-SAME: {
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:   rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x129x256x!qElemType, {order = #NHWC}>
    // CHECK: [[CONV1TILE0:%.+]] = VPU.NCE.Convolution([[CONV0TILE0]], %arg3, %arg1)
    // CHECK-SAME: {
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME: pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>,
    // CHECK-SAME:  rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} -> tensor<1x32x128x256x!qElemType, {order = #NHWC}>
    // CHECK: [[SLICETILE0:%.+]] = VPU.Slice [[CONV0TILE0]] [0, 0, 0, 0] [1, 32, 128, 256]
    // CHECK: [[ELTWISETILE0:%.+]] = VPU.NCE.Eltwise([[SLICETILE0]], [[CONV1TILE0]])
    // CHECK: [[SLICEARG0TILE1:%.+]] = VPU.Slice %arg0 [0, 0, 126, 0] [1, 16, 130, 256]
    // CHECK: [[CONV0TILE1:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE1]], %arg2, %arg1)
    // CHECK-SAME: {
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:   rawFilterShape = [32, 16, 3, 3],
   // CHECK-SAME:    strides = [1, 1]} -> tensor<1x32x129x256x!qElemType, {order = #NHWC}>
    // CHECK: [[CONV1TILE1:%.+]] = VPU.NCE.Convolution([[CONV0TILE1]], %arg3, %arg1)
    // CHECK-SAME: {
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:   pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:   rawFilterShape = [32, 32, 3, 3],
    // CHECK-SAME:   strides = [1, 1]} -> tensor<1x32x128x256x!qElemType, {order = #NHWC}>
    // CHECK: [[SLICETILE1:%.+]] = VPU.Slice [[CONV0TILE1]] [0, 0, 1, 0] [1, 32, 128, 256]
    // CHECK: [[ELTWISETILE1:%.+]] = VPU.NCE.Eltwise([[SLICETILE1]], [[CONV1TILE1]])
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISETILE0]], [[ELTWISETILE1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 128, 0]]} : tensor<1x32x128x256x!qElemType, {order = #NHWC}>, tensor<1x32x128x256x!qElemType, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
    // CHECK: return [[CONCAT]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VfTilingWithEltwiseAdjustOffset
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
// CHECK-SAME:      [[WT:%.+]]: tensor<32x1x1x4xsi32>
// CHECK-SAME:      [[W1:%.+]]: tensor<32x16x3x3xf16, {order = #NHWC}>
// CHECK-SAME:      [[W2:%.+]]: tensor<32x32x3x3xf16, {order = #NHWC}>
func.func @VfTilingWithEltwiseAdjustOffset(
            %arg0: tensor<1x16x128x128xf16, {order = #NHWC}>,
            %wt: tensor<32x1x1x4xsi32>, %weights_1: tensor<32x16x3x3xf16, {order = #NHWC}>,
            %weights_2: tensor<32x32x3x3xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}>  {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x128x128xf16, {order = #NHWC}>, %weights_1 as %arg2: tensor<32x16x3x3xf16, {order = #NHWC}>, %wt as %arg3: tensor<32x1x1x4xsi32>,
                            %weights_2 as %arg4: tensor<32x32x7x7xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x128x128xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} -> tensor<1x32x128x128xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 7, 7], strides = [1, 1]} -> tensor<1x32x128x128xf16, {order = #NHWC}>
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
    %cst = const.Declare tensor<16x1x1x4xsi32> = dense<[[[[0, 0, 1065353216, 0]]], [[[256, 0, 1065353216, 0]]], [[[512, 0, 1065353216, 0]]], [[[768, 0, 1065353216, 0]]], [[[1024, 0, 1065353216, 0]]], [[[1280, 0, 1065353216, 0]]], [[[1536, 0, 1065353216, 0]]], [[[1792, 0, 1065353216, 0]]], [[[2048, 0, 1065353216, 0]]], [[[2304, 0, 1065353216, 0]]], [[[2560, 0, 1065353216, 0]]], [[[2816, 0, 1065353216, 0]]], [[[3072, 0, 1065353216, 0]]], [[[3328, 0, 1065353216, 0]]], [[[3584, 0, 1065353216, 0]]], [[[3840, 0, 1065353216, 0]]]]> : tensor<16x1x1x4xsi32>
    %cst_0 = const.Declare tensor<1x32x49x61xi1, {order = #NHWC}> = dense<1> : tensor<1x32x49x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %cst_1 = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %0 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = 32 : i64} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    %1 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x24x30xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<1x32x49x61xi1, {order = #NHWC}>, %0 as %arg3: tensor<1x1x49x61xi32, {order = #NHWC}>, %cst_1 as %arg4: tensor<16x32x2x2xf16, {order = #NHWC}>, %cst as %arg5: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x48x60xf16, {order = #NHWC}> {
      %2 = VPU.GroupSparseTensor(%arg1, %arg2, %arg3) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>} -> !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>
      %3 = VPU.NCE.Convolution(%2, %arg4, %arg5) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 32, 2, 2], strides = [1, 1]} -> tensor<1x16x48x60xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %1 : tensor<1x16x48x60xf16, {order = #NHWC}>

    // CHECK: [[SET:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = 32 : i64} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    // CHECK: [[SLICE_ARG_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 12, 30]
    // CHECK: [[SLICE_CST_0:%.+]] = VPU.Slice %cst_0 [0, 0, 0, 0] [1, 32, 25, 61]
    // CHECK: [[SLICE_SET_0:%.+]] = VPU.Slice [[SET]] [0, 0, 0, 0] [1, 1, 25, 61]
    // CHECK: [[GST0:%.+]] = VPU.GroupSparseTensor([[SLICE_ARG_0]], [[SLICE_CST_0]], [[SLICE_SET_0]])
    // CHECK-SAME: {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2], offsets = [0, 0, 0, 0], sizes = [1, 32, 25, 61]>
    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[GST0]], %cst_1, %cst)
    // CHECK-SAME: tensor<1x16x24x60xf16, {order = #NHWC}>
    // CHECK: [[SLICE_ARG_1:%.+]] = VPU.Slice %arg0 [0, 0, 11, 0] [1, 32, 13, 30]
    // CHECK: [[SLICE_CST_1:%.+]] = VPU.Slice %cst_0 [0, 0, 24, 0] [1, 32, 25, 61]
    // CHECK: [[SLICE_SET_1:%.+]] = VPU.Slice [[SET]] [0, 0, 24, 0] [1, 1, 25, 61]
    // CHECK: [[GST1:%.+]] = VPU.GroupSparseTensor([[SLICE_ARG_1]], [[SLICE_CST_1]], [[SLICE_SET_1]])
    // CHECK-SAME: {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 2, 2], offsets = [0, 0, 2, 0], sizes = [1, 32, 25, 61]>
    // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[GST1]], %cst_1, %cst)
    // CHECK-SAME: tensor<1x16x24x60xf16, {order = #NHWC}>
    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV0]], [[CONV1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 24, 0]]}
    // CHECK: return [[CONCAT]] : tensor<1x16x48x60xf16, {order = #NHWC}>
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
