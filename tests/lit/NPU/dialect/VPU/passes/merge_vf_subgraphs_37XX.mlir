//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

// CHECK-LABEL: @BuildSubgraphEltwise
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>)
func.func @BuildSubgraphEltwise(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_3 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST]] as [[ARG_2:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<32x1x1x4xsi32>,
    //CHECK-SAME:                         [[CST_1]] as [[ARG_4:%[^:]+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME:                         attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 3]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[ARG_4]], [[ARG_3]])
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV0]], [[CONV1]])
    //CHECK:        VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

// CHECK-LABEL: @BuildSubgraphEltwiseWithViewLikeOpInput
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x16x512x256x!qElemType>)
func.func @BuildSubgraphEltwiseWithViewLikeOpInput(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %arg1: tensor<1x16x512x256x!qElemType>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.LayoutCast(%arg1) {dst_order = #NHWC} : tensor<1x16x512x256x!qElemType> -> tensor<1x16x512x256x!qElemType, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 32, 256, 256]} inputs(%0 : tensor<1x16x512x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    %2 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg3: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg4: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Convolution(%arg2, %arg3, %arg4)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    %3 = VPU.VerticalFusion (%1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %2 as %arg3: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg2, %arg3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %3 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK-DAG: [[WEIGHT:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    //CHECK-DAG: [[BIAS:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    //CHECK: [[LAYOUTCAST:%.+]] = VPU.LayoutCast([[ARG_1]]) {dst_order = #NHWC} : tensor<1x16x512x256x!qElemType> -> tensor<1x16x512x256x!qElemType, {order = #NHWC}>
    //CHECK: [[SHAPECAST:%.+]] = VPU.ShapeCast {shape = [1, 32, 256, 256]} inputs([[LAYOUTCAST]] : tensor<1x16x512x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_2:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:              [[WEIGHT]] as [[ARG_3:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:              [[BIAS]] as [[ARG_4:%[^:]+]]: tensor<32x1x1x4xsi32>,
    //CHECK-SAME:              [[SHAPECAST]] as [[ARG_5:%[^:]+]]: tensor<1x32x256x256x!qElemType, {order = #NHWC}>)
    //CHECK-SAME:              attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 3]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution([[ARG_2]], [[ARG_3]], [[ARG_4]])
    //CHECK:   [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG_5]], [[CONV]])
    //CHECK:   VPU.Yield [[ELTWISE]]
    //CHECK:   return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildOutCMXSubgraph
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x150x256xf16, {order = #NHWC}>)
func.func @BuildOutCMXSubgraph(%arg0: tensor<1x32x150x256xf16, {order = #NHWC}>) -> tensor<1x32x150x256xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x32x11x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x32x11x1xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x150x256xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<16x32x11x1xf16, {order = #NHWC}>, %cst_5 as %arg3: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x150x256xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 5 : i64, bottom = 5 : i64>, ppe = #VPU.PPEStub<>,
         rawFilterShape = [16, 32, 11, 1], strides = [1, 1]} : tensor<1x32x150x256xf16, {order = #NHWC}>, tensor<16x32x11x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x150x256xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x150x256xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x150x256xf16, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x150x256xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x150x256xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    return %1 : tensor<1x32x150x256xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<16x32x11x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x11x1xf16>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x32x150x256xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_1]] as [[ARG_2:%[^:]+]]: tensor<16x32x11x1xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_2]] as [[ARG_3:%[^:]+]]: tensor<16x1x1x4xsi32>,
    //CHECK-SAME:                         [[CST]] as [[ARG_4:%[^:]+]]: tensor<32x16x3x3xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_5:%[^:]+]]: tensor<32x1x1x4xsi32>)
    //CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 6, 1]} -> tensor<1x32x150x256xf16, {order = #NHWC}> {
    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[ARG_4]], [[ARG_5]])
    //CHECK:  VPU.Yield [[CONV1]]
    //CHECK: return [[VERTICAL_FUSION]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

func.func @BuildSubgraphNotConstInput(%arg0: tensor<1x48x256x16xf16, {order = #NHWC}>, %arg1: tensor<8x4096x40xf16>, %arg2: tensor<8x4096x40xf16>) -> tensor<1x48x256x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>

    %0 = VPU.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [8, 4096, 40, 1]} : tensor<8x4096x40xf16> -> tensor<8x4096x40x1xf16>
    %1 = VPU.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHCW, tilingStrategy = [1, 1, 3, 1]} : tensor<8x4096x40x1xf16> -> tensor<8x40x4096x1xf16>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [320, 4096, 1, 1]} : tensor<8x40x4096x1xf16> -> tensor<320x4096x1x1xf16>
    %3 = VPU.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<320x4096x1x1xf16> -> tensor<320x4096x1x1xf16, {order = #NHWC}>
    %4 = VPU.Slice %3 [0, 0, 0, 0] [48, 4096, 1, 1] : tensor<320x4096x1x1xf16, {order = #NHWC}> to tensor<48x4096x1x1xf16, {order = #NHWC}>
    %5 = VPU.AffineReshape(%arg2) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32768, 40, 1, 1]} : tensor<8x4096x40xf16> -> tensor<32768x40x1x1xf16>
    %6 = VPU.PermuteCast(%5) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<32768x40x1x1xf16> -> tensor<32768x40x1x1xf16, {order = #NHWC}>
    %7 = VPU.Expand(%6) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<32768x40x1x1xf16, {order = #NHWC}> -> tensor<32768x48x1x1xf16, {order = #NHWC}>
    %8 = VPU.Slice %7 [0, 0, 0, 0] [4096, 48, 1, 1] : tensor<32768x48x1x1xf16, {order = #NHWC}> to tensor<4096x48x1x1xf16, {order = #NHWC}>
    %9 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x256x16xf16, {order = #NHWC}>, %8 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x4096x256x16xf16, {order = #NHWC}> {
      %11 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x256x16xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x256x16xf16, {order = #NHWC}>
      %12 = VPU.SoftMax(%11) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x256x16xf16, {order = #NHWC}> -> tensor<1x4096x256x16xf16, {order = #NHWC}>
      VPU.Yield %12
   }

   %10 = VPU.VerticalFusion (%9 as %arg3: tensor<1x4096x256x16xf16, {order = #NHWC}>, %4 as %arg4: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 22, 1]} -> tensor<1x48x256x16xf16, {order = #NHWC}> {
      %11 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x256x16xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x256x16xf16, {order = #NHWC}>
      VPU.Yield %11
   }

   return %10: tensor<1x48x256x16xf16, {order = #NHWC}>

   //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]])
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]]
   //CHECK: VPU.Yield [[CONV1]]

   //CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x256x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PartialBuildWithWeightsConstraints
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x256x48x14xf16, {order = #NHWC}>)
func.func @PartialBuildWithWeightsConstraints(%arg0: tensor<1x256x48x14xf16, {order = #NHWC}>) -> tensor<1x256x48x14xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    %cst_1 = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<256x256x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst_1 as %arg2: tensor<256x256x3x3xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [256, 256, 3, 3], strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x3x3xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (
        %1 as %arg1: tensor<1x256x48x14xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
            -> tensor<1x256x48x14xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} : tensor<1x256x48x14xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x48x14xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %2 : tensor<1x256x48x14xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x1x1xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x256x3x3xf16>

    // %1 and %2 are merged. %0 is not merged to avoid too large weights size
    //CHECK: [[VF_0:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x256x48x14xf16, {order = #NHWC}>,
    //CHECK-SAME: [[CST_1]] as [[ARG_2:%[^:]+]]: tensor<256x256x3x3xf16, {order = #NHWC}>,
    //CHECK-SAME: [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x256x48x14xf16, {order = #NHWC}>
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])

    //CHECK: [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as [[ARG_1:%[^:]+]]: tensor<1x256x48x14xf16, {order = #NHWC}>,
    //CHECK-SAME: [[CST]] as [[ARG_2:%[^:]+]]: tensor<256x256x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME: [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<256x1x1x4xsi32>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 4, 1]}
    //CHECK-SAME: -> tensor<1x256x48x14xf16, {order = #NHWC}>
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK:    [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_1]], [[ARG_2]], [[ARG_3]])

    //CHECK: return [[VF_1]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @BuildSubgraphNotConstInputWithExpand
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x48x256x16xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<8x4096x40xf16>, [[ARG_2:%[^:]+]]: tensor<8x4096x40xf16>)
func.func @BuildSubgraphNotConstInputWithExpand(%arg0: tensor<1x48x256x16xf16, {order = #NHWC}>, %arg1: tensor<8x4096x40xf16>, %arg2: tensor<8x4096x40xf16>) -> tensor<1x48x256x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<4096x1x1x4xsi32> = dense<1> : tensor<4096x1x1x4xsi32>

    %0 = VPU.AffineReshape(%arg1) {dim_mapping = [[0], [1], [2, 3]], shape_value = [8, 4096, 40, 1]} : tensor<8x4096x40xf16> -> tensor<8x4096x40x1xf16>
    %1 = VPU.MemPermute(%0) {dst_order = #NCHW, mem_perm = #NHCW, tilingStrategy = [1, 1, 3, 1]} : tensor<8x4096x40x1xf16> -> tensor<8x40x4096x1xf16>
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3]], shape_value = [320, 4096, 1, 1]} : tensor<8x40x4096x1xf16> -> tensor<320x4096x1x1xf16>
    %3 = VPU.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<320x4096x1x1xf16> -> tensor<320x4096x1x1xf16, {order = #NHWC}>
    %4 = VPU.Slice %3 [0, 0, 0, 0] [40, 4096, 1, 1] : tensor<320x4096x1x1xf16, {order = #NHWC}> to tensor<40x4096x1x1xf16, {order = #NHWC}>
    %5 = VPU.Expand(%4) {pads_begin = [0, 0, 0, 0], pads_end = [8, 0, 0, 0]} : tensor<40x4096x1x1xf16, {order = #NHWC}> -> tensor<48x4096x1x1xf16, {order = #NHWC}>
    %6 = VPU.AffineReshape(%arg2) {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [32768, 40, 1, 1]} : tensor<8x4096x40xf16> -> tensor<32768x40x1x1xf16>
    %7 = VPU.PermuteCast(%6) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<32768x40x1x1xf16> -> tensor<32768x40x1x1xf16, {order = #NHWC}>
    %8 = VPU.Expand(%7) {pads_begin = [0, 0, 0, 0], pads_end = [0, 8, 0, 0]} : tensor<32768x40x1x1xf16, {order = #NHWC}> -> tensor<32768x48x1x1xf16, {order = #NHWC}>
    %9 = VPU.Slice %8 [0, 0, 0, 0] [4096, 48, 1, 1] : tensor<32768x48x1x1xf16, {order = #NHWC}> to tensor<4096x48x1x1xf16, {order = #NHWC}>
    %10 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x48x256x16xf16, {order = #NHWC}>, %9 as %arg4: tensor<4096x48x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<4096x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 29, 1]} -> tensor<1x4096x256x16xf16, {order = #NHWC}> {
      %12 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, rawFilterShape = [4096, 48, 1, 1], strides = [1, 1]} : tensor<1x48x256x16xf16, {order = #NHWC}>, tensor<4096x48x1x1xf16, {order = #NHWC}>, tensor<4096x1x1x4xsi32> -> tensor<1x4096x256x16xf16, {order = #NHWC}>
      %13 = VPU.SoftMax(%12) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x4096x256x16xf16, {order = #NHWC}> -> tensor<1x4096x256x16xf16, {order = #NHWC}>
      VPU.Yield %13
   }

   %11 = VPU.VerticalFusion (%10 as %arg3: tensor<1x4096x256x16xf16, {order = #NHWC}>, %5 as %arg4: tensor<48x4096x1x1xf16, {order = #NHWC}>, %cst as %arg5: tensor<48x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 22, 1]} -> tensor<1x48x256x16xf16, {order = #NHWC}> {
      %12 = VPU.NCE.Convolution(%arg3, %arg4, %arg5)
      {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
      ppe = #VPU.PPEStub<>, rawFilterShape = [48, 4096, 1, 1], strides = [1, 1]} : tensor<1x4096x256x16xf16, {order = #NHWC}>, tensor<48x4096x1x1xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x256x16xf16, {order = #NHWC}>
      VPU.Yield %12
   }

   return %11: tensor<1x48x256x16xf16, {order = #NHWC}>

   //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
   //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution(
   //CHECK: [[SOFTMAX:%.+]] = VPU.SoftMax([[CONV0]])
   //CHECK-NOT: VPU.VerticalFusion
   //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[SOFTMAX]]
   //CHECK: VPU.Yield [[CONV1]]

   //CHECK: return [[VERTICAL_FUSION]] : tensor<1x48x256x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @BuildSubgraphConvWithDepthToSpace
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x180x270xf16, {order = #NHWC}>)
func.func @BuildSubgraphConvWithDepthToSpace(%arg0: tensor<1x16x180x270xf16, {order = #NHWC}>) -> tensor<1x1x720x1080xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x5x5xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x180x270xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<16x16x5x5xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x16x180x270xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
        ppe = #VPU.PPEStub<>, rawFilterShape = [16, 16, 5, 5], strides = [1, 1]} : tensor<1x16x180x270xf16, {order = #NHWC}>, tensor<16x16x5x5xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x180x270xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x180x270xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x1x720x1080xf16, {order = #NHWC}> {
      %3 = VPU.DepthToSpace(%arg1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x16x180x270xf16, {order = #NHWC}> -> tensor<1x1x720x1080xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %1 : tensor<1x1x720x1080xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x16x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x5x5xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x180x270xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST]] as [[ARG_2:%[^:]+]]: tensor<16x16x5x5xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME:                         attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 2]} -> tensor<1x1x720x1080xf16, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK:      [[D2S:%.+]] = VPU.DepthToSpace([[CONV0]])
    //CHECK:        VPU.Yield [[D2S]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x1x720x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PartialBuildWithLargestOpMemoryConstraints
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x180x270xf16, {order = #NHWC}>)
func.func @PartialBuildWithLargestOpMemoryConstraints(%arg0: tensor<1x32x180x270xf16, {order = #NHWC}>) -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
    %cst_6 = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x32x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_7 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x16x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_38 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %cst_39 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x180x270xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<16x32x1x1xf16, {order = #NHWC}>, %cst_39 as %arg3: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x16x180x270xf16, {order = #NHWC}> {
      %98 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, rawFilterShape = [16, 32, 1, 1], strides = [1, 1]} : tensor<1x32x180x270xf16, {order = #NHWC}>, tensor<16x32x1x1xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x180x270xf16, {order = #NHWC}>
      VPU.Yield %98
    }

    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x180x270xf16, {order = #NHWC}>, %cst_7 as %arg2: tensor<128x16x1x1xf16, {order = #NHWC}>, %cst_38 as %arg3: tensor<128x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 5, 1]} -> tensor<1x128x180x270xf16, {order = #NHWC}> {
      %98 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>, rawFilterShape = [128, 16, 1, 1], strides = [1, 1]} : tensor<1x16x180x270xf16, {order = #NHWC}>, tensor<128x16x1x1xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x180x270xf16, {order = #NHWC}>
      VPU.Yield %98
    }

    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x128x180x270xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
      %98 = VPU.DepthToSpace(%arg1) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
      VPU.Yield %98
    }

    return %2 : tensor<1x8x720x1080xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x32x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x1x1xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x16x1x1xf16>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:      [[VF_0:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x32x180x270xf16, {order = #NHWC}>,
    //CHECK-SAME:              [[CST]] as [[ARG_2:%[^:]+]]: tensor<16x32x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:              [[CST_2]] as [[ARG_3:%[^:]+]]: tensor<16x1x1x4xsi32>,
    //CHECK-SAME:              [[CST_0]] as [[ARG_4:%[^:]+]]: tensor<128x16x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:              [[CST_1]] as [[ARG_5:%[^:]+]]: tensor<128x1x1x4xsi32>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 9, 1]} -> tensor<1x128x180x270xf16, {order = #NHWC}> {
    //CHECK:      [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK:      [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[ARG_4]], [[ARG_5]])
    //CHECK:        VPU.Yield [[CONV_1]]

    //CHECK:      [[VF_1:%.+]] = VPU.VerticalFusion ([[VF_0]] as [[ARG_1:%[^:]+]]: tensor<1x128x180x270xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 15, 1]} -> tensor<1x8x720x1080xf16, {order = #NHWC}> {
    //CHECK:      [[D2S:%.+]] = VPU.DepthToSpace([[ARG_1]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x128x180x270xf16, {order = #NHWC}> -> tensor<1x8x720x1080xf16, {order = #NHWC}>
    //CHECK:        VPU.Yield [[D2S]]

    //CHECK: return [[VF_1]] : tensor<1x8x720x1080xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType1 = !quant.uniform<u8:f16, 1.6479117599188113>
!qElemType2 = !quant.uniform<u8:f16, 7.0412104587928921>

// CHECK-LABEL: @BuildSubgraphWithSameInputsEltwise
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x96x180x320x!qElemType, {order = #NHWC}>)
func.func @BuildSubgraphWithSameInputsEltwise(%arg0: tensor<1x96x180x320x!qElemType1, {order = #NHWC}>) -> tensor<1x96x180x320x!qElemType2, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x96x180x320x!qElemType1, {order = #NHWC}>, %arg0 as %arg2: tensor<1x96x180x320x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 6, 1]} -> tensor<1x96x180x320xf16, {order = #NHWC}> {
      %445 = VPU.NCE.Eltwise(%arg2, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x96x180x320xf16, {order = #NHWC}>
      VPU.Yield %445
    }

    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x96x180x320xf16, {order = #NHWC}>, %0 as %arg2: tensor<1x96x180x320xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 8, 1]} -> tensor<1x96x180x320x!qElemType2, {order = #NHWC}> {
      %445 = VPU.NCE.Eltwise(%arg2, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x96x180x320x!quant.uniform<u8:f16, 7.0412104587928921>, {order = #NHWC}>
      VPU.Yield %445
    }

    return  %1 : tensor<1x96x180x320x!qElemType2, {order = #NHWC}>

    //CHECK:      [[VF_0:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x96x180x320x!qElemType, {order = #NHWC}>, [[ARG_0]] as [[ARG_2:%[^:]+]]: tensor<1x96x180x320x!qElemType, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 10, 1]} -> tensor<1x96x180x320x!qElemType1, {order = #NHWC}> {
    //CHECK:      [[ADD_0:%.+]] = VPU.NCE.Eltwise([[ARG_2]], [[ARG_2]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:          ppe = #VPU.PPEStub<>} -> tensor<1x96x180x320xf16, {order = #NHWC}>
    //CHECK:      [[ADD_1:%.+]] = VPU.NCE.Eltwise([[ADD_0]], [[ADD_0]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:          ppe = #VPU.PPEStub<>} -> tensor<1x96x180x320x!qElemType1, {order = #NHWC}>
    //CHECK:       VPU.Yield [[ADD_1]]
    //CHECK:      }
    //CHECK:      return [[VF_0]] : tensor<1x96x180x320x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubgraphOneParentNotVF
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x416x416xf16, {order = #NHWC}>, [[ARG_1:%[^:]+]]: tensor<1x32x208x208xf16, {order = #NHWC}>)
func.func @BuildSubgraphOneParentNotVF(%arg0: tensor<1x32x416x416xf16, {order = #NHWC}>, %arg1: tensor<1x32x208x208xf16, {order = #NHWC}>) -> tensor<1x64x208x208xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst_1)
       {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
       pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
       ppe = #VPU.PPEStub<>, rawFilterShape = [64, 32, 3, 3], strides = [2, 2], tilingStrategy = [1, 1, 8, 1]}
       : tensor<1x32x416x416xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x208x208xf16, {order = #NHWC}>
    %1 = VPU.VerticalFusion (%arg1 as %arg2: tensor<1x32x208x208xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x32x3x3xf16, {order = #NHWC}>, %cst_1 as %arg4: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg2, %arg3, %arg4)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>, rawFilterShape = [64, 32, 3, 3], strides = [1, 1]} : tensor<1x32x208x208xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x208x208xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (%0 as %arg2: tensor<1x64x208x208xf16, {order = #NHWC}>, %1 as %arg3: tensor<1x64x208x208xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg2, %arg3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
         -> tensor<1x64x208x208xf16, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2: tensor<1x64x208x208xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    //CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_0]]
    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_1]] as [[ARG_2:%[^:]+]]: tensor<1x32x208x208xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST]] as [[ARG_3:%[^:]+]]: tensor<64x32x3x3xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_4:%[^:]+]]: tensor<64x1x1x4xsi32>,
    //CHECK-SAME:                         [[CONV0]] as [[ARG_5:%[^:]+]]: tensor<1x64x208x208xf16, {order = #NHWC}>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 6, 1]} -> tensor<1x64x208x208xf16, {order = #NHWC}> {
    //CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[ARG_2]], [[ARG_3]], [[ARG_4]])
    //CHECK: [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[ARG_5]], [[CONV1]])
    //CHECK:  VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x64x208x208xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildSubgraphCTiling(%arg0: tensor<1x128x64x96xf16, {order = #NHWC}>) -> tensor<1x128x64x96xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x64x96xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x64x96xf16, {order = #NHWC}> {
      %2 = VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x128x64x96xf16, {order = #NHWC}> -> tensor<1x128x64x96xf16, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x128x64x96xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x64x96xf16, {order = #NHWC}> {
      %2 = VPU.NCE.AveragePool(%arg1) {kernel_size = [1, 1], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, strides = [1, 1]} -> tensor<1x128x64x96xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %1 : tensor<1x128x64x96xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:              attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 2, 1, 1]}
    //CHECK:   [[MVN:%.+]] = VPU.MVN
    //CHECK:   [[AVG:%.+]] = VPU.NCE.AveragePool([[MVN]])
    //CHECK:   VPU.Yield [[AVG]]
    //CHECK:   return [[VERTICAL_FUSION]] : tensor<1x128x64x96xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.066607063891840915:126>
!qElemType1 = !quant.uniform<u8<0:254>:f16:0, {0.0012998153844217615:127,0.0018982229035670363:127,0.0019103605446853036:127,0.0016835428129030963:127,0.001929748715378168:127,0.0013972403496269165:127,0.0019231087814165851:127,0.0017214799959828534:127,0.0019708599631241925:127,0.0014245941882997048:127,0.0015013835092229167:127,0.0018210335979311485:127,0.0019365317943527943:127,0.0013182708832222645:127,0.001946882352115601:127,0.001452652957495742:127,0.001253475823740321:127,0.0016627796287611715:127,0.0013371993472256999:127,0.0017889444752940981:127,0.0014539933580113209:127,0.0020158159451221856:127,0.0013332571101000929:127,0.0016296942402997355:127,0.0018043224736461489:127,0.0013885323222227923:127,0.0014750117392051878:127,0.001251295443594925:127,0.0017561241397707481:127,0.001258520277466361:127,0.0012454000983651229:127,0.0019671725710545939:127,0.0013832205862510862:127,0.0014796034088284951:127,0.0016176862510170523:127,0.0013194100593957375:127,0.0012687479886483019:127,0.0016104801902620811:127,0.001808305190304133:127,0.001686601422903106:127,0.0014129187178424025:127,0.0013911974007689107:127,0.0018313568173431037:127,0.0020283010062270277:127,0.0013118773464142806:127,0.0015647336253969688:127,0.0018739950234495748:127,0.0013380488307457271:127,0.0019991081061325675:127,0.0016516142004118191:127,0.0015377592383407233:127,0.0012948443805138896:127,0.0020322393713973637:127,0.0014817999807868417:127,0.0013128348926859578:127,0.0014753593938557181:127,0.0014060409519616075:127,0.0017390227693272389:127,0.0020264896351521408:127,0.0016461690579812358:127,0.0014954381805705273:127,0.0015151248438151803:127,0.0017349283526262899:127,0.0012640091847247025:127}>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotBuildSubgraphDiffTiling(%arg0: tensor<1x64x128x384x!qElemType, {order = #NHWC}>) -> tensor<1x64x128x384xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x64x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x128x384x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<64x64x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x128x384x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x128x384x!qElemType, {order = #NHWC}>, tensor<64x64x3x3x!qElemType1, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x128x384x!qElemType, {order = #NHWC}>
      VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x64x128x384xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x64x128x384xf16, {order = #NHWC}> {
      %2 = VPU.MVN(%arg1) {across_channels = false, eps = 9.9999997473787516E-6 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, normalize_variance = true} : tensor<1x64x128x384xf16, {order = #NHWC}> -> tensor<1x64x128x384xf16, {order = #NHWC}>
      VPU.Yield %2
    }

    return %1 : tensor<1x64x128x384xf16, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:              attributes {tilingStrategy = [1, 1, 2, 1]}
    //CHECK:   [[CONV:%.+]] = VPU.NCE.Convolution
    //CHECK:   VPU.Yield [[CONV]]
    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:              attributes {tilingStrategy = [1, 2, 1, 1]}
    //CHECK:   [[MVN:%.+]] = VPU.MVN
    //CHECK:   VPU.Yield [[MVN]]
    //CHECK:   return [[VERTICAL_FUSION1]] : tensor<1x64x128x384xf16, {order = #NHWC}>
}

// -----

!qElemType0 = !quant.uniform<u8:f16, 0.066607063891840915:126>
!qElemType1 = !quant.uniform<u8:f16, 0.066607063891840915:127>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @NotBuildSubgraphInaccurateTiling(%arg0: tensor<1x384x16x48x!qElemType0, {order = #NHWC}>) -> tensor<1x384x16x48x!qElemType0, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<384x384x1x1x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<384x384x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<384x1x1x4xsi32> = dense<1> : tensor<384x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x384x16x48x!qElemType0, {order = #NHWC}>, %cst_0 as %arg2: tensor<384x384x1x1x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<384x1x1x4xsi32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x384x16x48x!qElemType0, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [384, 384, 1, 1], strides = [1, 1]} : tensor<1x384x16x48x!qElemType0, {order = #NHWC}>, tensor<384x384x1x1x!qElemType1, {order = #NHWC}>, tensor<384x1x1x4xsi32> -> tensor<1x384x16x48x!qElemType0, {order = #NHWC}>
      VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x384x16x48x!qElemType0, {order = #NHWC}>, %cst_0 as %arg2: tensor<384x384x1x1x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<384x1x1x4xsi32>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x384x16x48x!qElemType0, {order = #NHWC}>  {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [384, 384, 1, 1], strides = [1, 1]} : tensor<1x384x16x48x!qElemType0, {order = #NHWC}>, tensor<384x384x1x1x!qElemType1, {order = #NHWC}>, tensor<384x1x1x4xsi32> -> tensor<1x384x16x48x!qElemType0, {order = #NHWC}>
      VPU.Yield %2
    }

    return %1 : tensor<1x384x16x48x!qElemType0, {order = #NHWC}>

    //CHECK: [[VERTICAL_FUSION0:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:              attributes {tilingStrategy = [1, 2, 1, 1]}
    //CHECK:   [[CONV0:%.+]] = VPU.NCE.Convolution
    //CHECK:   VPU.Yield [[CONV0]]
    //CHECK: [[VERTICAL_FUSION1:%.+]] = VPU.VerticalFusion
    //CHECK-SAME:              attributes {tilingStrategy = [1, 2, 1, 1]}
    //CHECK:   [[CONV1:%.+]] = VPU.NCE.Convolution
    //CHECK:   VPU.Yield [[CONV1]]
    //CHECK:   return [[VERTICAL_FUSION1]] : tensor<1x384x16x48x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubgraphWithViewLikeOp
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x32x24x30xf16, {order = #NHWC}>)
func.func @BuildSubgraphWithViewLikeOp(%arg0: tensor<1x32x24x30xf16, {order = #NHWC}>) -> tensor<1x16x48x60xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x32x24x30xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x24x30xf16, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg1, %arg1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x24x30xf16, {order = #NHWC}>
      VPU.Yield %4
    }
    %cst = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %1 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    %cst_0 = const.Declare tensor<1x32x49x61xi1, {order = #NHWC}> = dense<1> : tensor<1x32x49x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %cst_1 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x24x30xf16, {order = #NHWC}>, %cst_0 as %arg2: tensor<1x32x49x61xi1, {order = #NHWC}>, %1 as %arg3: tensor<1x1x49x61xi32, {order = #NHWC}>, %cst as %arg4: tensor<16x32x2x2xf16, {order = #NHWC}>, %cst_1 as %arg5: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x48x60xf16, {order = #NHWC}> {
      %3 = VPU.GroupSparseTensor(%arg1, %arg2, %arg3) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>} -> !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>
      %4 = VPU.NCE.Convolution(%3, %arg4, %arg5) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 32, 2, 2], strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>, tensor<16x32x2x2xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x48x60xf16, {order = #NHWC}>
      VPU.Yield %4
    }
    return %2 : tensor<1x16x48x60xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<1x32x49x61xi1, {order = #NHWC}> = dense<1> : tensor<1x32x49x61xi8>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>

    //CHECK:  [[SET:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    //CHECK:  VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x32x24x30xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_2:%[^:]+]]: tensor<1x32x49x61xi1, {order = #NHWC}>,
    //CHECK-SAME:                         [[SET]] as [[ARG_3:%[^:]+]]: tensor<1x1x49x61xi32, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_1]] as [[ARG_4:%[^:]+]]: tensor<16x32x2x2xf16, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST]] as [[ARG_5:%[^:]+]]: tensor<16x1x1x4xsi32>)
    //CHECK-SAME: attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 2, 1]}
    //CHECK:  [[ELT:%.+]] = VPU.NCE.Eltwise([[ARG_1]], [[ARG_1]])
    //CHECK:  [[GST:%.+]] = VPU.GroupSparseTensor([[ELT]], [[ARG_2]], [[ARG_3]])
    //CHECK:  [[CONV:%.+]] = VPU.NCE.Convolution([[GST]], [[ARG_4]], [[ARG_5]])
    //CHECK:  VPU.Yield [[CONV]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

// CHECK-LABEL: @BuildTooLargeSubgraph
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>)
func.func @BuildTooLargeSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%2, %3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                                               [[CST]] as [[ARG_2:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:                                               [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<32x1x1x4xsi32>,
    //CHECK-SAME:                                               [[CST_1]] as [[ARG_4:%[^:]+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME:                                               attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 3]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[ARG_4]], [[ARG_3]])
    //CHECK:      [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV1]], [[ARG_4]], [[ARG_3]])
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], [[CONV2]])
    //CHECK:      VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>
!qElemType2 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128}>

// CHECK-LABEL: @BuildLargeSubgraph
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>)
func.func @BuildLargeSubgraph(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<16x16x1x1x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_4 as %arg2: tensor<16x16x1x1x!qElemType2, {order = #NHWC}>, %cst_5 as %arg3: tensor<16x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         rawFilterShape = [16, 16, 1, 1], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<16x16x1x1x!qElemType2, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>, %cst_2 as %arg4: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg4, %arg3)
         {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
         ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %4 = VPU.NCE.Eltwise(%2, %3)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }
    return %1 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    // CHECK-DAG:   [[CST_1:%.+]] = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>
    // CHECK-DAG:   [[CST_2:%.+]] = const.Declare tensor<16x16x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>
    // CHECK-DAG:   [[CST_3:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_2]] as [[ARG_2:%[^:]+]]: tensor<16x16x1x1x!qElemType2, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_3]] as [[ARG_3:%[^:]+]]: tensor<16x1x1x4xsi32>,
    //CHECK-SAME:                         [[CST]] as [[ARG_4:%[^:]+]]: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>,
    //CHECK-SAME:                         [[CST_0]] as [[ARG_5:%[^:]+]]: tensor<32x1x1x4xsi32>,
    //CHECK-SAME:                         [[CST_1]] as [[ARG_6:%[^:]+]]: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME:                         attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 3, 1]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]])
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]], [[ARG_4]], [[ARG_5]])
    //CHECK:      [[CONV2:%.+]] = VPU.NCE.Convolution([[CONV1]], [[ARG_6]], [[ARG_5]])
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV1]], [[CONV2]])
    //CHECK:      VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x32x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @BuildSubgraphConvSwishGroupConvVF
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x176x176xf16, {order = #NHWC}>)
func.func @BuildSubgraphConvSwishGroupConvVF(%arg0: tensor<1x16x176x176xf16, {order = #NHWC}>) -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>
    %cst_2 = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
          %arg0 as %arg1: tensor<1x16x176x176xf16, {order = #NHWC}>,
          %cst_0 as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
          %cst_1 as %arg3: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 3, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %2 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
          {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]}
             : tensor<1x16x176x176xf16, {order = #NHWC}>, tensor<96x16x1x1xf16, {order = #NHWC}>, tensor<96x1x1x4xsi32> -> tensor<1x96x176x176xf16, {order = #NHWC}>
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
       %cst_2 as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
       %cst_3 as %arg3: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
          -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %4 = VPU.NCE.DepthConvolution(
       %arg1, %arg2, %arg3) {
       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
       ppe = #VPU.PPEStub<>,
       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
       rawFilterShape = [96, 1, 1, 1],
       strides = [1, 1]
       } -> tensor<1x96x176x176xf16, {order = #NHWC}>
      VPU.Yield %4
    }

    return %2 : tensor<1x96x176x176xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x16x176x176xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[CST]] as [[ARG_2:%[^:]+]]: tensor<96x16x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<96x1x1x4xsi32>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 7, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[ARG_2]], [[ARG_3]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 16, 1, 1], strides = [1, 1]}
    //CHECK-SAME:      -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[SWISH0:%.+]] = VPU.Swish([[CONV0]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], [[ARG_2]], [[ARG_3]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:        VPU.Yield [[DWCONV0]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x96x176x176xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeSubgraphsWithCompatibleDistributedTensorType
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x96x176x176xf16, {order = #NHWC}>)
func.func @MergeSubgraphsWithCompatibleDistributedTensorType(%arg0: tensor<1x96x176x176xf16, {order = #NHWC}>) -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    %0 = VPU.VerticalFusion (
        %arg0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
        %2 = VPU.Swish(%arg1)
            {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
            tensor<1x96x176x176xf16, {order = #NHWC}>
                -> tensor<1x96x176x176xf16, {order = #NHWC}>
        VPU.Yield %2
    }
    %1 = VPU.VerticalFusion (
        %0 as %arg1: tensor<1x96x176x176xf16, {order = #NHWC}>,
        %cst as %arg2: tensor<96x16x1x1xf16, {order = #NHWC}>,
        %cst_0 as %arg3: tensor<96x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]}
             -> tensor<1x96x176x176xf16, {order = #NHWC}> {
        %3 = VPU.NCE.DepthConvolution(
            %arg1, %arg2, %arg3) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [96, 1, 1, 1],
            strides = [1, 1]
        } -> tensor<1x96x176x176xf16, {order = #NHWC}>
        VPU.Yield %3
    }

    return %1 : tensor<1x96x176x176xf16, {order = #NHWC}>

    // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<96x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<96x16x1x1xf16>
    // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<96x1x1x4xsi32> = dense<1> : tensor<96x1x1x4xsi32>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion ([[ARG_0]] as [[ARG_1:%[^:]+]]: tensor<1x96x176x176xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[CST]] as [[ARG_2:%[^:]+]]: tensor<96x16x1x1xf16, {order = #NHWC}>,
    //CHECK-SAME:                        [[CST_0]] as [[ARG_3:%[^:]+]]: tensor<96x1x1x4xsi32>) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 7, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}> {
    //CHECK:      [[SWISH0:%.+]] = VPU.Swish([[ARG_1]]) {beta_value = 1.000000e+00 : f64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x96x176x176xf16, {order = #NHWC}> -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:      [[DWCONV0:%.+]] = VPU.NCE.DepthConvolution([[SWISH0]], [[ARG_2]], [[ARG_3]]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [96, 1, 1, 1], strides = [1, 1]} -> tensor<1x96x176x176xf16, {order = #NHWC}>
    //CHECK:        VPU.Yield [[DWCONV0]]

    //CHECK: return [[VERTICAL_FUSION]] : tensor<1x96x176x176xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00:8>
!qElemType1 = !quant.uniform<u8:f16, 0.081176862529679844:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @HandleSpatialDim1Ops
// CHECK-SAME:  [[INPUT0:%.+]]: tensor<16384x2048x1x1x!qElemType, {order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x2048x1x1xf16, {order = #NHWC}>
func.func @HandleSpatialDim1Ops(%arg0: tensor<16384x2048x1x1x!qElemType, {order = #NHWC}>,
                                 %arg1: tensor<1x2048x1x1xf16, {order = #NHWC}>) -> tensor<1x16384x1x1x!qElemType1, {order = #NHWC}> {
    %weightsTable = const.Declare tensor<16384x1x1x4xsi32> = dense<1> : tensor<16384x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<16384x2048x1x1x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [15, 1, 1, 1]} -> tensor<16384x2048x1x1xf16, {order = #NHWC}> {
        %2 = VPU.Dequantize(%arg2) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<16384x2048x1x1x!qElemType, {order = #NHWC}> -> tensor<16384x2048x1x1xf16, {order = #NHWC}>
        VPU.Yield %2
    }

    %1 = VPU.VerticalFusion (%arg1 as %arg2: tensor<1x2048x1x1xf16, {order = #NHWC}>, %0 as %arg3: tensor<16384x2048x1x1xf16, {order = #NHWC}>, %weightsTable as %arg4: tensor<16384x1x1x4xsi32>) attributes {tilingStrategy = [1, 64, 1, 1]} -> tensor<1x16384x1x1x!qElemType1, {order = #NHWC}> {
        %2 = VPU.NCE.Convolution(%arg2, %arg3, %arg4) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16384, 2048, 1, 1], strides = [1, 1]} : tensor<1x2048x1x1xf16, {order = #NHWC}>, tensor<16384x2048x1x1xf16, {order = #NHWC}>, tensor<16384x1x1x4xsi32> -> tensor<1x16384x1x1x!qElemType1, {order = #NHWC}>
        VPU.Yield %2
    }

    return %1 : tensor<1x16384x1x1x!qElemType1, {order = #NHWC}>

    // CHECK: [[VF:%.+]] = VPU.VerticalFusion
    // CHECK-SAME: scenario = #VPU.vf_scenario<FULL_PREFETCHING>
    // CHECK-SAME: tilingStrategy = [1, 64, 1, 1]
    // CHECK:     VPU.Dequantize
    // CHECK:     VPU.NCE.Convolution
    // CHECK:     VPU.Yield

    // CHECK: return [[VF]]
}
