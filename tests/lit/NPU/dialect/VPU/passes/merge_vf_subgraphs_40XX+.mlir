//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --merge-vertical-fusion-subgraphs %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @BuildSubgraphEltwise(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>


    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ({{[^ ]+}} as [[INPUT:%.+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                         {{[^ ]+}} as [[WEIGHTS_0:%.+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, {{[^ ]+}} as [[WEIGHTS_1:%.+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_0]])
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[WEIGHTS_1]])
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV0]], [[CONV1]])
    //CHECK:        VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128}>

func.func @BuildLargeSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x16x1x1x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_4 as %arg2: tensor<16x16x1x1x!qElemType2, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType2, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_2 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%2, %3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ({{[^ ]+}} as [[INPUT:%.+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                         {{[^ ]+}} as [[WEIGHTS_0:%.+]]: tensor<16x16x1x1x!qElemType2, {order = #NHWC}>, {{[^ ]+}} as [[WEIGHTS_1:%.+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:                         {{[^ ]+}} as [[WEIGHTS_2:%.+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME:                         attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_0]])
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[WEIGHTS_1]])
    //CHECK:      [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV1]], [[WEIGHTS_2]])
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], [[CONV2]])
    //CHECK:      VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @BuildSubgraphEltwiseWithViewLikeOpInput(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %arg1: tensor<1x16x512x256x!qElemType>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]

    %0 = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x512x256x!qElemType> -> tensor<1x16x512x256x!qElemType, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 256, 256]} inputs(%0 : tensor<1x16x512x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    %2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    %3 = VPU.VerticalFusion (%1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %2 as %arg3: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg2, %arg3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHT:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]

    //CHECK: [[LAYOUTCAST:%.+]] = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x512x256x!qElemType> -> tensor<1x16x512x256x!qElemType, {order = #NHWC}>
    //CHECK: [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 256, 256]} inputs(%0 : tensor<1x16x512x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ({{[^ ]+}} as [[INPUT:%.+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:              [[WEIGHT]] as [[CONV_WEIGHT:%.+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:              [[SHAPECAST]] as [[ELTWISE_INPUT:%.+]]: tensor<1x32x256x256x!qElemType, {order = #NHWC}>)
    //CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[CONV_WEIGHT]])
    //CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ELTWISE_INPUT]], [[CONV]])
    //CHECK:   VPU.Yield [[ELTWISE]]
    //CHECK:   return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildSubgraphConvSwishGroupConvVF(%arg0: tensor<1x16x176x176xf16, {order = #NHWC}>) -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
          %arg0 as %arg1: tensor<1x16x176x176xf16, {order = #NHWC}>,
          %cst_0 as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>
          ) attributes {tilingStrategy = [1, 1, 3, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %2 = VPU.NCE.Convolution(%arg1, %arg2)
          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]}
             : tensor<1x16x176x176xf16, {order = #NHWC}>, tensor<96x16x1x1xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (
         %0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %3 = VPU.Swish(%arg1)
         {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
         tensor<1x96x176x176xf16, {order = #NHWC}>
            -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (
       %1 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>,
       %cst_2 as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]}
          -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %4 = VPU.NCE.DepthConvolution(
       %arg1, %arg2) {
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
       ppe = #VPU.PPEStub<>,
       rawFilterShape = [96, 1, 1, 1],
       strides = [1, 1]
       } -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %4
    }

    return %2 : tensor<1x96x176x176xf16, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x176x176xf16, {order = #NHWC}>,
    //CHECK-SAME:                        %cst as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME:                        scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[SWISH0:%.+]] = VPU.Swish([[CONV0]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:        VPU.Yield [[DWCONV0]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x96x176x176xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>
!qElemType2 = !quant.uniform<u8:f16, 0.013744638480392158:128>

func.func @BuildSubgraphVFInput(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %arg1: tensor<1x16x256x256x!qElemType2, {order = #NHWC}>) -> (tensor<1x32x256x256x!qElemType, {order = #NHWC}>) {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg1 as %arg2: tensor<1x16x256x256x!qElemType2, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType2, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType2, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType2, {order = #NHWC}>
      VPU.Yield %4
    }
    %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType} : tensor<1x32x256x256x!qElemType2, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    %3 = VPU.VerticalFusion (%2 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %1 as %arg3: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(%arg2, %arg3)
    //CHECK: VPU.Yield [[CONV0]]

    //CHECK: [[QUANTCAST:%.+]] = VPU.QuantizeCast([[VERTICAL_FUSION0]]) {dstElemType = !qElemType}
    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution(%arg2, %arg3)
    //CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], %arg4)
    //CHECK: VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION1]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphConvGeluVF
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @BuildSubgraphConvGeluVF(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16xf32>, [#const.Reshape<[64, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x1x1xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
      %3 = VPU.Gelu(%arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x64x128x128xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16xf32>, [#const.Reshape<[64, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[WEIGHTS]] as [[INNER_ARG2:[^:]+]]: tensor<64x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                        ) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    //CHECK: [[NCE:%.+]] = VPU.NCE.Convolution([[INNER_ARG1]], [[INNER_ARG2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]}
    //CHECK-SAME: -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK: [[GELU:%.+]] = VPU.Gelu([[NCE]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK:  VPU.Yield [[GELU]]
    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x64x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphConvSigmoidVF
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @BuildSubgraphConvSigmoidVF(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x1x1xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
      %3 = VPU.Sigmoid(%arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x64x128x128xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[WEIGHTS]] as [[INNER_ARG2:[^:]+]]: tensor<64x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                        ) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    //CHECK: [[NCE:%.+]] = VPU.NCE.Convolution([[INNER_ARG1]], [[INNER_ARG2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]}
    //CHECK-SAME: -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK: [[SIGMOID:%.+]] = VPU.Sigmoid([[NCE]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK:  VPU.Yield [[SIGMOID]]
    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x64x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.025219376414429909:55>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

func.func @NotBuildOutOfLimitNumTiles(%arg0: tensor<1x512x92x120x!qElemType, {order = #NHWC}>) -> tensor<1x256x92x120x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<256x512x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<256x512x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<256x256x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<256x256x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x512x92x120x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<256x512x3x3x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 92, 1]} -> tensor<1x256x92x120x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
          ppe = #VPU.PPEStub<>,
          rawFilterShape = [256, 512, 3, 3], strides = [1, 1]} : tensor<1x512x92x120x!qElemType, {order = #NHWC}>, tensor<256x512x3x3x!qElemType, {order = #NHWC}> -> tensor<1x256x92x120x!qElemType, {order = #NHWC}>
      VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x256x92x120x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<256x256x3x3x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 16, 1]} -> tensor<1x256x92x120x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2)
          {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
           ppe = #VPU.PPEStub<>,
           rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} : tensor<1x256x92x120x!qElemType, {order = #NHWC}>, tensor<256x256x3x3x!qElemType, {order = #NHWC}> -> tensor<1x256x92x120x!qElemType, {order = #NHWC}>
      VPU.Yield %2
    }

    return %1 : tensor<1x256x92x120x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 92, 1]

    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 16, 1]

    //CHECK: return [[VERTICAL_FUSION1]]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @PartialBuildWithWeightsConstraints(%arg0: tensor<1x256x48x14xf16, {order = #NHWC}>) -> tensor<1x256x48x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst_1 as %arg2: tensor<256x256x3x3xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [256, 256, 3, 3], ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x3x3xf16, {order = #NHWC}> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
         {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [256, 256, 1, 1], ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (
        %1 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
         {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [256, 256, 1, 1], ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %2 : tensor<1x256x48x14xf16, {order = #NHWC}>

    // %1 and %2 are merged. %0 is not merged to avoid too large weights size
    //CHECK: [[VF_0:%.+]] = VPU.VerticalFusion ({{[^ ]+}} as [[INPUT:%.+]]: tensor<1x256x48x14xf16, {order = #NHWC}>,
    //CHECK-SAME: {{[^ ]+}} as [[WEIGHTS:%.+]]: tensor<256x256x3x3xf16, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 2, 1]}
    //CHECK-SAME: -> tensor<1x256x48x14xf16, {order = #NHWC}>
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])

    //CHECK: [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as [[INPUT:%.+]]: tensor<1x256x48x14xf16, {order = #NHWC}>,
    //CHECK-SAME: {{[^ ]+}} as [[WEIGHTS:%.+]]: tensor<256x256x1x1xf16, {order = #NHWC}>)
    //CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]}
    //CHECK-SAME: -> tensor<1x256x48x14xf16, {order = #NHWC}>
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
    //CHECK:    [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_1]], [[WEIGHTS]])

    //CHECK: return [[VF_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildSubgraphConvWithDepthToSpace(%arg0: tensor<1x16x180x270xf16, {order = #NHWC}>) -> tensor<1x1x720x1080xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x5x5xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x180x270xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<16x16x5x5xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x180x270xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
      pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
      ppe = #VPU.PPEStub<>,
      rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : tensor<1x16x180x270xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}> -> tensor<1x16x180x270xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x180x270xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x1x720x1080xf16, {order = #NHWC}> {
      %3 = VPU.DepthToSpace(%arg1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x180x270xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x1x720x1080xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
      VPU.Yield %3
    }

    return %1 : tensor<1x1x720x1080xf16, {order = #NHWC}>


    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x180x270xf16, {order = #NHWC}>,
    //CHECK-SAME:                         %cst as %arg2: tensor<16x16x5x5xf16, {order = #NHWC}>)
    //CHECK-SAME:                         attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x1x720x1080xf16, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution(%arg1, %arg2)
    //CHECK:      [[D2S:%.+]] = VPU.DepthToSpace([[CONV0]])
    //CHECK:        VPU.Yield [[D2S]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x1x720x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildMultiplySoftmaxSubgraph(%arg0: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1: tensor<1x4x160x160xf16, {order = #NHWC}>) -> tensor<1x4x160x160xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
      %3 = VPU.Multiply(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
      %3 = VPU.SoftMax(%arg2) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x4x160x160xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
    //CHECK: [[MUL:%.+]] = VPU.Multiply(%arg2, %arg3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
    //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[MUL]]) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
    //CHECK:  VPU.Yield [[SOFTMAX]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x4x160x160xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @BuildSWAddSoftmaxSubgraph
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x32x16x16xf16>, [[INPUT1:%.+]]: tensor<1x1x16x16xf16>
func.func @BuildSWAddSoftmaxSubgraph(%arg0: tensor<1x32x16x16xf16>, %arg1: tensor<1x1x16x16xf16>) -> tensor<1x32x16x16xf16> {

    %0 = VPU.VerticalFusion (%arg0 as %arg22: tensor<1x32x16x16xf16>, %arg1 as %arg23: tensor<1x1x16x16xf16>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x16x16xf16> {
      %570 = VPU.Add(%arg22, %arg23) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x32x16x16xf16>
      VPU.Yield %570
    }
    %1 = VPU.VerticalFusion (%0 as %arg22: tensor<1x32x16x16xf16>) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x32x16x16xf16> {
      %570 = VPU.SoftMax(%arg22) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16xf16>
      VPU.Yield %570
    }

    return %1: tensor<1x32x16x16xf16>

    // CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:[^:]+]]: tensor<1x32x16x16xf16>, [[INPUT1]] as [[INNER_ARG1:[^:]+]]: tensor<1x1x16x16xf16>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x16x16xf16> {
    // CHECK: [[ADD:%.+]] = VPU.Add([[INNER_ARG0]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x16x16xf16>, tensor<1x1x16x16xf16> -> tensor<1x32x16x16xf16>
    // CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[ADD]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x16x16xf16> -> tensor<1x32x16x16xf16>
    // CHECK:  VPU.Yield [[SOFTMAX]]

    // CHECK: return [[VERTICAL_FUSION]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0094078685723099058:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0047039342861549529:128>

// CHECK-LABEL: @BuildSubgraphEltwiseQuantizeCastVF
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x32x48x48x!qElemType, {order = #NHWC}>
func.func @BuildSubgraphEltwiseQuantizeCastVF(%arg0: tensor<1x32x48x48x!qElemType, {order = #NHWC}>) -> tensor<1x32x48x48xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x48x48x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x48x48x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Eltwise(%arg1, %arg1)
         {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x48x48x!qElemType, {order = #NHWC}>
      VPU.Yield %2
  }

  %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x32x48x48x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x48x48xf16, {order = #NHWC}> {
      %3 = VPU.QuantizeCast(%arg2) {dstElemType = !qElemType1} : tensor<1x32x48x48x!qElemType, {order = #NHWC}> -> tensor<1x32x48x48x!qElemType1, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%3, %3)
         {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x48x48xf16, {order = #NHWC}>
      VPU.Yield %4
  }
  return %1 : tensor<1x32x48x48xf16, {order = #NHWC}>

  //CHECK:  VPU.VerticalFusion ([[INPUT]] as %arg1: tensor<1x32x48x48x!qElemType, {order = #NHWC}>)
  //CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]}
  //CHECK:  [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise(%arg1, %arg1)
  //CHECK:  [[QUANTIZE_CAST:%.+]] = VPU.QuantizeCast([[ELTWISE_0]])
  //CHECK:  [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[QUANTIZE_CAST]], [[QUANTIZE_CAST]])
  //CHECK:  VPU.Yield [[ELTWISE_1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphConvAbsVF
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @BuildSubgraphConvAbsVF(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16xf32>, [#const.Reshape<[64, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<64x16x1x1xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x64x128x128xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
      %3 = VPU.Abs(%arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x64x128x128xf16, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16xf32>, [#const.Reshape<[64, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[WEIGHTS]] as [[INNER_ARG2:[^:]+]]: tensor<64x16x1x1xf16, {order = #NHWC}>
    //CHECK-SAME:                        ) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x128xf16, {order = #NHWC}> {
    //CHECK: [[NCE:%.+]] = VPU.NCE.Convolution([[INNER_ARG1]], [[INNER_ARG2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 1, 1], strides = [1, 1]}
    //CHECK-SAME: -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK: [[ABS:%.+]] = VPU.Abs([[NCE]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x128x128xf16, {order = #NHWC}> -> tensor<1x64x128x128xf16, {order = #NHWC}>
    //CHECK:  VPU.Yield [[ABS]]
    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x64x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @NotBuildNotPipelinedSubgraph(%arg0: tensor<1x128x32x64xf16, {order = #NHWC}>) -> tensor<1x64x64x128xf16> {
    %cst_0 = const.Declare tensor<1x128x67x131xi1, {order = #NHWC}> = dense<1> : tensor<1x128x67x131xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %cst_1 = const.Declare tensor<64x128x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %cst_3 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %set = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 128, 32, 64], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = [128]} -> tensor<1x1x67x131xi32, {order = #NHWC}>
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x32x64xf16, {order = #NHWC}>, %cst_3 as %arg2: tensor<128x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x128x32x64xf16, {order = #NHWC}> {
      %2 = VPU.NCE.DepthConvolution(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [128, 1, 1, 1], strides = [1, 1]} -> tensor<1x128x32x64xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x128x32x64xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<1x128x67x131xi1, {order = #NHWC}>, %set as %arg3: tensor<1x1x67x131xi32, {order = #NHWC}>, %cst_1 as %arg4: tensor<64x128x4x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x64x128xf16> {
      %2 = VPU.GroupSparseTensor(%arg1, %arg2, %arg3) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>} -> !VPU.SparseTensor<data=tensor<1x128x32x64xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x67x131xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x67x131xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>
      %3 = VPU.NCE.Convolution(%2, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 128, 4, 4], strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x128x32x64xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x67x131xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x67x131xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>, tensor<64x128x4x4xf16, {order = #NHWC}> -> tensor<1x64x64x128xf16>
      VPU.Yield %3
    }

    return %1  : tensor<1x64x64x128xf16>

    //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion (%arg0
    //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]]

    //CHECK: return [[VF_1]]  : tensor<1x64x64x128xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @NotBuildSOKDuplicatedSubgraph(%arg0: tensor<1x32x1x5375xf16, {order = #NHWC}>) -> tensor<1x384x1x5375xf16> {
    %cst_0 = const.Declare tensor<384x1x1x128xi1> = dense<1.0> : tensor<384x32x1x1xf16, {order = #NHWC}>, [#const.GetSparsityMap]
    %cst_1 = const.Declare tensor<384x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<384x32x1x1xf16, {order = #NHWC}>, [#const.Sparsify<false>]

    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_4 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.Reorder<#NHWC>]

    %gst = VPU.GroupSparseTensor(%cst_1, %cst_0) {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<24> : tensor<384xi64>, alignment = 16 : i64>} -> !VPU.SparseTensor<data=tensor<384x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<384x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<24> : tensor<384xi64>, alignment = 16 : i64>>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x32x1x5375xf16, {order = #NHWC}>, %cst_4 as %arg3: tensor<32x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x1x5375xf16, {order = #NHWC}> {
      %3 = VPU.NCE.DepthConvolution(%arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x1x5375xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x32x1x5375xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x1x5375xf16, {order = #NHWC}> {
      %3 = VPU.Gelu(%arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1x5375xf16, {order = #NHWC}> -> tensor<1x32x1x5375xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (%1 as %arg2: tensor<1x32x1x5375xf16, {order = #NHWC}>, %gst as %arg3: !VPU.SparseTensor<data=tensor<384x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<384x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<24> : tensor<384xi64>, alignment = 16 : i64>>) attributes {tilingStrategy = [1, 4, 1, 1]} -> tensor<1x384x1x5375xf16> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [384, 32, 1, 1], strides = [1, 1]} : tensor<1x32x1x5375xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<384x32x1x1xf16, {order = #NHWC}>, sparsity_map=tensor<384x1x1x128xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<24> : tensor<384xi64>, alignment = 16 : i64>> -> tensor<1x384x1x5375xf16>
      VPU.Yield %3
    }

    return %2  : tensor<1x384x1x5375xf16>

    //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion (%arg0
    //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]]
    //CHECK:  [[VF_2:%.+]] = VPU.VerticalFusion ([[VF_1]]

    //CHECK: return [[VF_2]]  : tensor<1x384x1x5375xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphConvPReluVF
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @BuildSubgraphConvPReluVF(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x128xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16xf32>, [#const.Reshape<[16, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<16x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x128x128xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_2 as %arg3: tensor<1x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x128x128xf16, {order = #NHWC}> {
      %3 = VPU.PRelu(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x16x128x128xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16xf32>, [#const.Reshape<[16, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG: [[CST:%.+]] = const.Declare
    // CHECK-SAME:       tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]

    // CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:                        [[WEIGHTS]] as [[INNER_ARG2:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                        [[CST]] as [[INNER_ARG4:[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x128x128xf16, {order = #NHWC}> {
    // CHECK: [[NCE:%.+]] = VPU.NCE.Convolution([[INNER_ARG1]], [[INNER_ARG2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]}
    // CHECK-SAME: -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[NCE]], [[INNER_ARG4]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
    // CHECK-SAME:       tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:  VPU.Yield [[PRELU]]
    // CHECK: return [[VERTICAL_FUSION]] : tensor<1x16x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @BuildSubgraphConvPReluVFWithDifferentTiles
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>
func.func @BuildSubgraphConvPReluVFWithDifferentTiles(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x128xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16xf32>, [#const.Reshape<[16, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<16x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x128x128xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x16x128x128xf16, {order = #NHWC}>, %cst_2 as %arg3: tensor<1x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x128x128xf16, {order = #NHWC}> {
      %3 = VPU.PRelu(%arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x16x128x128xf16, {order = #NHWC}>

    // CHECK-DAG: [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16xf32>, [#const.Reshape<[16, 16, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    // CHECK-DAG: [[CST:%.+]] = const.Declare
    // CHECK-SAME:       tensor<1x16x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf16>, [#const.Reshape<[1, 5, 1, 1]>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]

    // CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG1:[^:]+]]: tensor<1x16x128x128xf16, {order = #NHWC}>,
    // CHECK-SAME:                        [[WEIGHTS]] as [[INNER_ARG2:[^:]+]]: tensor<16x16x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:                        [[CST]] as [[INNER_ARG4:[^:]+]]: tensor<1x16x1x1xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x128x128xf16, {order = #NHWC}> {
    // CHECK: [[NCE:%.+]] = VPU.NCE.Convolution([[INNER_ARG1]], [[INNER_ARG2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 1, 1], strides = [1, 1]}
    // CHECK-SAME: -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK: [[PRELU:%.+]] = VPU.PRelu([[NCE]], [[INNER_ARG4]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
    // CHECK-SAME:       tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x128x128xf16, {order = #NHWC}>
    // CHECK:  VPU.Yield [[PRELU]]
    // CHECK: return [[VERTICAL_FUSION]] : tensor<1x16x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

//CHECK-LABEL: @DoNotMergeOverlappedSW
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x256x48x14xf16, {order = #NHWC}>
func.func @DoNotMergeOverlappedSW(%arg0: tensor<1x256x48x14xf16, {order = #NHWC}>) -> tensor<1x256x48x14xf16, {order = #NHWC}> {
    %cst_1 = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]}
             -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.Swish(%arg1)
          {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
          tensor<1x256x48x14xf16, {order = #NHWC}>
              -> tensor<1x256x48x14xf16, {order = #NHWC}>
        VPU.Yield %3
    }

    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst_1 as %arg2: tensor<256x256x3x3xf16, {order = #NHWC}>
        ) attributes {tilingStrategy = [1, 1, 2, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [256, 256, 3, 3], ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x3x3xf16, {order = #NHWC}> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1 : tensor<1x256x48x14xf16, {order = #NHWC}>

    //CHECK:  [[VF_0:%.+]] = VPU.VerticalFusion ([[INPUT]]
    //CHECK:  [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]]

    //CHECK: return [[VF_1]]  : tensor<1x256x48x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubtractMultiplySubgraph
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x4x160x160xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x4x160x160xf16, {order = #NHWC}>)
func.func @BuildSubtractMultiplySubgraph(%arg0: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1: tensor<1x4x160x160xf16, {order = #NHWC}>) -> tensor<1x4x160x160xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
      %3 = VPU.Multiply(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg2: tensor<1x4x160x160xf16, {order = #NHWC}>, %arg1 as %arg3: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
      %3 = VPU.Subtract(%arg2, %arg3)
         {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1: tensor<1x4x160x160xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT0]] as [[INNER_ARG0:%.+]]: tensor<1x4x160x160xf16, {order = #NHWC}>, [[INPUT1]] as [[INNER_ARG1:%.+]]: tensor<1x4x160x160xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 4, 1]} -> tensor<1x4x160x160xf16, {order = #NHWC}> {
    //CHECK: [[MUL:%.+]] = VPU.Multiply([[INNER_ARG0]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
    //CHECK: [[SUB:%.+]] = VPU.Subtract([[MUL]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4x160x160xf16, {order = #NHWC}>, tensor<1x4x160x160xf16, {order = #NHWC}> -> tensor<1x4x160x160xf16, {order = #NHWC}>
    //CHECK:  VPU.Yield [[SUB]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x4x160x160xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

//CHECK-LABEL: @BuildSubgraphMemPermuteClustered
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x16x256x256xf16, {order = #NWCH}>
func.func @BuildSubgraphMemPermuteClustered(%arg0: tensor<1x16x256x256xf16, {order = #NWCH}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16, {order = #NHWC}>
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256xf16, {order = #NWCH}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x16x256x256xf16, {order = #NHWC}> {
      %3 = VPU.MemPermute(%arg1)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         dst_order = #NHWC, mem_perm = #NWCH} : tensor<1x16x256x256xf16, {order = #NWCH}> -> tensor<1x16x256x256xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x256x256xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 1, 1], strides = [1, 1]} : tensor<1x16x256x256xf16, {order = #NHWC}>, tensor<32x16x1x1xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %1 : tensor<1x32x256x256xf16, {order = #NHWC}>
    // CHECK:      [[CST_0:%.+]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    // CHECK:          [[INPUT]] as %arg1
    // CHECK:          [[CST_0]] as [[CONV_WEIGHTS:%[^:]+]]
    // CHECK:      [[PERMUTE:%.+]] = VPU.MemPermute
    // CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[PERMUTE]], [[CONV_WEIGHTS]])
    // CHECK:      return [[VERTICAL_FUSION]] : tensor<1x32x256x256xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @BuildNormalizeL2MultiplySubgraph
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x60x801x384xf16>
func.func @BuildNormalizeL2MultiplySubgraph(%arg0: tensor<1x60x801x384xf16>) -> tensor<1x60x801x384xf16> {
    %cst = const.Declare tensor<1x1x1x384xf16> = dense<2.0> : tensor<1x1x1x384xf16>
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x60x801x384xf16>) attributes {tilingStrategy = [1, 1, 13, 1]} -> tensor<1x60x801x384xf16> {
      %2 = VPU.NormalizeL2(%arg1) {axes_value = [3], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x60x801x384xf16> -> tensor<1x60x801x384xf16>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x60x801x384xf16>, %cst as %arg2: tensor<1x1x1x384xf16>) attributes {tilingStrategy = [1, 1, 20, 1]} -> tensor<1x60x801x384xf16> {
      %2 = VPU.Multiply(%arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x60x801x384xf16>, tensor<1x1x1x384xf16> -> tensor<1x60x801x384xf16>
      VPU.Yield %2
    }

    return %1: tensor<1x60x801x384xf16>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1x1x1x384xf16> = dense<2.000000e+00> : tensor<1x1x1x384xf16>
    // CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[INPUT]] as [[INNER_ARG0:%.+]]: tensor<1x60x801x384xf16>, [[CST]] as [[INNER_ARG1:%.+]]: tensor<1x1x1x384xf16>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>,
    // CHECK: [[NORMALIZE_L2:%.+]] = VPU.NormalizeL2([[INNER_ARG0]]) {axes_value = [3], eps = 9.999999960041972E-13 : f64, eps_mode = #IE.eps_mode<ADD>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x60x801x384xf16> -> tensor<1x60x801x384xf16>
    // CHECK: [[MUL:%.+]] = VPU.Multiply([[NORMALIZE_L2]], [[INNER_ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>,  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x60x801x384xf16>, tensor<1x1x1x384xf16> -> tensor<1x60x801x384xf16>
    // CHECK:  VPU.Yield [[MUL]]

    // CHECK: return [[VERTICAL_FUSION]] : tensor<1x60x801x384xf16>
}
