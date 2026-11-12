//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --vertical-fusion-tiling %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>
!qElemType1 = !quant.uniform<u8:f16, 0.026811079885445389:79>
!qElemType2 = !quant.uniform<u8<0:254>:f16:0, {0.0015777572402803917:127,0.0014889139359391581:127,0.0021083918143445114:127,0.0019076836156094168:127,0.002125650878966324:127,0.0015942034289592832:127,0.0013417831556064876:127,0.0013941065298290704:127,0.0015372968330158022:127,0.0022980439381336602:127,0.0019483130982541662:127,0.0013906989745267732:127,0.0019419781101031566:127,0.001397803895116791:127,0.0016154645700154342:127,0.0019699626081571804:127,0.0020237670639368494:127,0.0015820100551515114:127,0.002136513708144661:127,0.0015983497063944659:127,0.0013165938572620782:127,0.0023659495856818251:127,0.0020228220721868078:127,0.0014110315503097894:127,0.0016812501460548462:127,0.0013487547870696061:127,0.0017490208618284211:127,0.0016929984796704269:127,0.0023835024495763102:127,0.0016608983278274536:127,0.0014221570857866543:127,0.0020535398186661128:127}>
!qElemType3 = !quant.uniform<u8<0:254>:f16:0, {0.0050877895880871871:127,0.0036900820225242554:127,0.0029946140886291744:127,0.0046036679913678503:127,0.0030745821674977702:127,0.0040151899255166839:127,0.0026243648191136637:127,0.002185353378611287:127,0.0053012159865672192:127,0.0019454215689906924:127,0.0043949132829200566:127,0.0021944085913380299:127,0.0045137020546620289:127,0.0027246975053952433:127,0.0026398010141267551:127,0.0035128525392277036:127,0.0055316268928407688:127,0.0037249932138938603:127,0.0031018390899568093:127,0.0031733273521183042:127,0.0048826055263909767:127,0.0055176643874701552:127,0.0026252196999046748:127,0.0023261205417903388:127,0.0029696068895144726:127,0.0052650167247441813:127,0.0054816298597440945:127,0.0024053254934746451:127,0.0044391516625411865:127,0.0043663044614116039:127,0.0028751257836349365:127,0.0051064467805577076:127}>
!qElemType4 = !quant.uniform<u8<0:254>:f16:0, {0.0021634723727158673:127,0.0025575409724017768:127,0.0038781130877066786:127,0.0032656551815393401:127,0.0020378891407974121:127,0.0035840688258644165:127,0.0028856976295080711:127,0.0023135664894824892:127,0.0034765291401720423:127,0.0033485044644573541:127,0.0035138308532594695:127,0.0018750902001313336:127,0.0018750902001313336:127,0.0018750902001313336:127,0.0018750902001313336:127,0.0018750902001313336:127}>
!qElemType5 = !quant.uniform<u8:f16, 0.016268887239343981:73>
!qElemType6 = !quant.uniform<u8:f16, 0.011832303860608269:83>
!qElemType7 = !quant.uniform<u8:f16, 0.020894650851978974:128>

// CHECK-LABEL: @TileEltwiseWithTwoVFLinkedInputs
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>
func.func @TileEltwiseWithTwoVFLinkedInputs(%arg0: tensor<1x16x256x256x!qElemType1, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {

   %cst_4 = const.Declare tensor<16x32x3x3x!qElemType4, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType4>, #const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<32x32x3x3x!qElemType3, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType6>, #const.Reorder<#NHWC>]
   %cst_6 = const.Declare tensor<32x16x3x3x!qElemType2, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]


   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType1, {order = #NHWC}>, %cst_6 as %arg2: tensor<32x16x3x3x!qElemType2, {order = #NHWC}>, %cst_5 as %arg4: tensor<32x32x3x3x!qElemType3, {order = #NHWC}>, %cst_4 as %arg6: tensor<16x32x3x3x!qElemType4, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 6, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x16x256x256x!qElemType1, {order = #NHWC}>, tensor<32x16x3x3x!qElemType2, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType5, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x32x256x256x!qElemType5, {order = #NHWC}>, tensor<32x32x3x3x!qElemType3, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType6, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType7, {order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %arg6) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x32x256x256x!qElemType7, {order = #NHWC}>, tensor<16x32x3x3x!qElemType4, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %0 : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK-DAG: [[WEIGHTS0:%.+]] = const.Declare tensor<16x32x3x3x
    // CHECK-DAG: [[WEIGHTS1:%.+]] = const.Declare tensor<32x32x3x3x
    // CHECK-DAG: [[WEIGHTS2:%.+]] = const.Declare tensor<32x16x3x3x
    // CHECK: [[SLICEARG0TILE0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 16, 46, 256]
    // CHECK: [[CONV0TILE0:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE0]], [[WEIGHTS2]])
    // CHECK: [[CONV1TILE0:%.+]] = VPU.NCE.Convolution([[CONV0TILE0]], [[WEIGHTS1]])
    // CHECK: [[SLICETILE0:%.+]] = VPU.Slice [[CONV0TILE0]] [0, 0, 0, 0] [1, 32, 44, 256]
    // CHECK: [[ELTWISETILE0:%.+]] = VPU.NCE.Eltwise([[SLICETILE0]], [[CONV1TILE0]])
    // CHECK: [[CONV2TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISETILE0]], [[WEIGHTS0]])
    // CHECK: [[SLICEARG0TILE1:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 40, 0] [1, 16, 49, 256]
    // CHECK: [[CONV0TILE1:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE1]], [[WEIGHTS2]])
    // CHECK: [[CONV1TILE1:%.+]] = VPU.NCE.Convolution([[CONV0TILE1]], [[WEIGHTS1]])
    // CHECK: [[SLICETILE1:%.+]] = VPU.Slice [[CONV0TILE1]] [0, 0, 1, 0] [1, 32, 45, 256]
    // CHECK: [[ELTWISETILE1:%.+]] = VPU.NCE.Eltwise([[SLICETILE1]], [[CONV1TILE1]])
    // CHECK: [[CONV2TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISETILE1]], [[WEIGHTS0]])
    // CHECK: [[SLICEARG0TILE2:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 83, 0] [1, 16, 49, 256]
    // CHECK: [[CONV0TILE2:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE2]], [[WEIGHTS2]])
    // CHECK: [[CONV1TILE2:%.+]] = VPU.NCE.Convolution([[CONV0TILE2]], [[WEIGHTS1]])
    // CHECK: [[SLICETILE2:%.+]] = VPU.Slice [[CONV0TILE2]] [0, 0, 1, 0] [1, 32, 45, 256]
    // CHECK: [[ELTWISETILE2:%.+]] = VPU.NCE.Eltwise([[SLICETILE2]], [[CONV1TILE2]])
    // CHECK: [[CONV2TILE2:%.+]] = VPU.NCE.Convolution([[ELTWISETILE2]], [[WEIGHTS0]])
    // CHECK: [[SLICEARG0TILE3:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 126, 0] [1, 16, 49, 256]
    // CHECK: [[CONV0TILE3:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE3]], [[WEIGHTS2]])
    // CHECK: [[CONV1TILE3:%.+]] = VPU.NCE.Convolution([[CONV0TILE3]], [[WEIGHTS1]])
    // CHECK: [[SLICETILE3:%.+]] = VPU.Slice [[CONV0TILE3]] [0, 0, 1, 0] [1, 32, 45, 256]
    // CHECK: [[ELTWISETILE3:%.+]] = VPU.NCE.Eltwise([[SLICETILE3]], [[CONV1TILE3]])
    // CHECK: [[CONV2TILE3:%.+]] = VPU.NCE.Convolution([[ELTWISETILE3]], [[WEIGHTS0]])
    // CHECK: [[SLICEARG0TILE4:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 169, 0] [1, 16, 48, 256]
    // CHECK: [[CONV0TILE4:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE4]], [[WEIGHTS2]])
    // CHECK: [[CONV1TILE4:%.+]] = VPU.NCE.Convolution([[CONV0TILE4]], [[WEIGHTS1]])
    // CHECK: [[SLICETILE4:%.+]] = VPU.Slice [[CONV0TILE4]] [0, 0, 1, 0] [1, 32, 44, 256]
    // CHECK: [[ELTWISETILE4:%.+]] = VPU.NCE.Eltwise([[SLICETILE4]], [[CONV1TILE4]])
    // CHECK: [[CONV2TILE4:%.+]]  = VPU.NCE.Convolution([[ELTWISETILE4]], [[WEIGHTS0]])
    // CHECK: [[SLICEARG0TILE5:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 211, 0] [1, 16, 45, 256]
    // CHECK: [[CONV0TILE5:%.+]] = VPU.NCE.Convolution([[SLICEARG0TILE5]], [[WEIGHTS2]])
    // CHECK: [[CONV1TILE5:%.+]] = VPU.NCE.Convolution([[CONV0TILE5]], [[WEIGHTS1]])
    // CHECK: [[SLICETILE5:%.+]] = VPU.Slice [[CONV0TILE5]] [0, 0, 1, 0] [1, 32, 43, 256]
    // CHECK: [[ELTWISETILE5:%.+]] = VPU.NCE.Eltwise([[SLICETILE5]], [[CONV1TILE5]])
    // CHECK: [[CONV2TILE5:%.+]] = VPU.NCE.Convolution([[ELTWISETILE5]], [[WEIGHTS0]])

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[CONV2TILE0]], [[CONV2TILE1]], [[CONV2TILE2]], [[CONV2TILE3]], [[CONV2TILE4]], [[CONV2TILE5]])
    // CHECK-SAME: {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]]}

    // CHECK: return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileEltwiseChain
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x56x56xf16, {order = #NHWC}>)
func.func @TileEltwiseChain(%arg0: tensor<1x64x56x56xf16, {order = #NHWC}>) -> tensor<1x256x56x56xf16, {order = #NHWC}> {

   %cst_3 = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.Reorder<#NHWC>]
   %cst_4 = const.Declare tensor<256x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x64x1x1xf16>, [#const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<64x256x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x256x1x1xf16>, [#const.Reorder<#NHWC>]
   %cst_6 = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]


   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x56x56xf16, {order = #NHWC}>,
                            %cst_4 as %arg2: tensor<256x64x1x1xf16, {order = #NHWC}>,
                            %cst_6 as %arg4: tensor<64x64x1x1xf16, {order = #NHWC}>,
                            %cst_3 as %arg6: tensor<64x64x3x3xf16, {order = #NHWC}>,
                            %cst_4 as %arg8: tensor<256x64x1x1xf16, {order = #NHWC}>,
                            %cst_5 as %arg10: tensor<64x256x1x1xf16, {order = #NHWC}>,
                            %cst_3 as %arg12: tensor<64x64x3x3xf16, {order = #NHWC}>,
                            %cst_4 as %arg14: tensor<256x64x1x1xf16, {order = #NHWC}>,
                            %cst_5 as %arg16: tensor<64x256x1x1xf16, {order = #NHWC}>,
                            %cst_3 as %arg18: tensor<64x64x3x3xf16, {order = #NHWC}>,
                            %cst_4 as %arg20: tensor<256x64x1x1xf16, {order = #NHWC}>
                            )
   attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x256x56x56xf16, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [256, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<256x64x1x1xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%arg1, %arg4) rawFilterShape [64, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg6) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %arg8) rawFilterShape [256, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<256x64x1x1xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%4, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x256x56x56xf16, {order = #NHWC}>
      %6 = VPU.NCE.Convolution(%5, %arg10) rawFilterShape [64, 256, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<64x256x1x1xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>
      %7 = VPU.NCE.Convolution(%6, %arg12) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>
      %8 = VPU.NCE.Convolution(%7, %arg14) rawFilterShape [256, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<256x64x1x1xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>
      %9 = VPU.NCE.Eltwise(%8, %5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x256x56x56xf16, {order = #NHWC}>
      %10 = VPU.NCE.Convolution(%9, %arg16) rawFilterShape [64, 256, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x256x56x56xf16, {order = #NHWC}>, tensor<64x256x1x1xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>
      %11 = VPU.NCE.Convolution(%10, %arg18) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}> -> tensor<1x64x56x56xf16, {order = #NHWC}>
      %12 = VPU.NCE.Convolution(%11, %arg20) rawFilterShape [256, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x56x56xf16, {order = #NHWC}>, tensor<256x64x1x1xf16, {order = #NHWC}> -> tensor<1x256x56x56xf16, {order = #NHWC}>
      %13 = VPU.NCE.Eltwise(%12, %9) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x256x56x56xf16, {order = #NHWC}>
      VPU.Yield %13
    }


    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>


    // CHECK: [[SLICE0TILE0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 64, 56, 30]
    // CHECK: [[CONV0TILE0:%.+]] = VPU.NCE.Convolution([[SLICE0TILE0]]
    // CHECK: [[SLICE1ARG0TILE0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 64, 56, 31]
    // CHECK: [[CONV1TILE0:%.+]] = VPU.NCE.Convolution([[SLICE1ARG0TILE0]],
    // CHECK: [[CONV2TILE0:%.+]] = VPU.NCE.Convolution([[CONV1TILE0]],
    // CHECK: [[CONV3TILE0:%.+]] = VPU.NCE.Convolution([[CONV2TILE0]],
    // CHECK: [[ELTWISE0TILE0:%.+]] = VPU.NCE.Eltwise([[CONV3TILE0]], [[CONV0TILE0]])
    // CHECK: [[CONV4TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE0TILE0]],
    // CHECK: [[CONV5TILE0:%.+]] = VPU.NCE.Convolution([[CONV4TILE0]],
    // CHECK: [[CONV6TILE0:%.+]] = VPU.NCE.Convolution([[CONV5TILE0]],
    // CHECK: [[SLICE0TILE0:%.+]] = VPU.Slice [[ELTWISE0TILE0]] [0, 0, 0, 0] [1, 256, 56, 29]
    // CHECK: [[ELTWISE1TILE0:%.+]] = VPU.NCE.Eltwise([[CONV6TILE0]], [[SLICE0TILE0]])
    // CHECK: [[CONV7TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE1TILE0]],
    // CHECK: [[CONV8TILE0:%.+]] = VPU.NCE.Convolution([[CONV7TILE0]],
    // CHECK: [[CONV9TILE0:%.+]] = VPU.NCE.Convolution([[CONV8TILE0]],
    // CHECK: [[SLICE1TILE0:%.+]] = VPU.Slice [[ELTWISE1TILE0]] [0, 0, 0, 0] [1, 256, 56, 28]
    // CHECK: [[ELTWISE2TILE0:%.+]] = VPU.NCE.Eltwise([[CONV9TILE0]], [[SLICE1TILE0]])

    // CHECK: [[SLICE0TILE1:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 26] [1, 64, 56, 30]
    // CHECK: [[CONV0TILE1:%.+]] = VPU.NCE.Convolution([[SLICE0TILE1]]
    // CHECK: [[SLICE1ARG0TILE1:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 25] [1, 64, 56, 31]
    // CHECK: [[CONV1TILE1:%.+]] = VPU.NCE.Convolution([[SLICE1ARG0TILE1]],
    // CHECK: [[CONV2TILE1:%.+]] = VPU.NCE.Convolution([[CONV1TILE1]],
    // CHECK: [[CONV3TILE1:%.+]] = VPU.NCE.Convolution([[CONV2TILE1]],
    // CHECK: [[ELTWISE0TILE1:%.+]] = VPU.NCE.Eltwise([[CONV3TILE1]], [[CONV0TILE1]])
    // CHECK: [[CONV4TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE0TILE1]],
    // CHECK: [[CONV5TILE1:%.+]] = VPU.NCE.Convolution([[CONV4TILE1]],
    // CHECK: [[CONV6TILE1:%.+]] = VPU.NCE.Convolution([[CONV5TILE1]],
    // CHECK: [[SLICE0TILE1:%.+]] = VPU.Slice [[ELTWISE0TILE1]] [0, 0, 0, 1] [1, 256, 56, 29]
    // CHECK: [[ELTWISE1TILE1:%.+]] = VPU.NCE.Eltwise([[CONV6TILE1]], [[SLICE0TILE1]])
    // CHECK: [[CONV7TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE1TILE1]],
    // CHECK: [[CONV8TILE1:%.+]] = VPU.NCE.Convolution([[CONV7TILE1]],
    // CHECK: [[CONV9TILE1:%.+]] = VPU.NCE.Convolution([[CONV8TILE1]],
    // CHECK: [[SLICE1TILE1:%.+]] = VPU.Slice [[ELTWISE1TILE1]] [0, 0, 0, 1] [1, 256, 56, 28]
    // CHECK: [[ELTWISE2TILE1:%.+]] = VPU.NCE.Eltwise([[CONV9TILE1]], [[SLICE1TILE1]])


    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[ELTWISE2TILE0]], [[ELTWISE2TILE1]])
    // CHECK-SAME: {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 28]]}

    // CHECK: return [[CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @main
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x128x68x120xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<1x32x272x480xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<1x64x136x240xf16, {order = #NHWC}>)
func.func @main(%arg0: tensor<1x128x68x120xf16, {order = #NHWC}>, %arg1: tensor<1x32x272x480xf16, {order = #NHWC}>,
%arg2: tensor<1x64x136x240xf16, {order = #NHWC}>) -> tensor<1x32x272x480xf16, {order = #NHWC}> {

    %cst_1 = const.Declare tensor<1x64x275x483xi1, {order = #NHWC}> = dense<1> : tensor<1x64x275x483xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %cst_3 = const.Declare tensor<1x128x139x243xi1, {order = #NHWC}> = dense<1> : tensor<1x128x139x243xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]

    %cst_4 = const.Declare tensor<64x128x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<128x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<128x64x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_6 = const.Declare tensor<64x128x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<64x128x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_7 = const.Declare tensor<32x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_8 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_9 = const.Declare tensor<32x64x4x4xf16, {order = #NHWC}> = dense<1.0> : tensor<32x64x4x4xf16>, [#const.Reorder<#NHWC>]
    %cst_10 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %storage_0 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 128, 68, 120], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = [128]} -> tensor<1x1x139x243xi32, {order = #NHWC}>
   %storage_1 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 64, 136, 240], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>, seDepth = 1 : i64, seSize = [64]} -> tensor<1x1x275x483xi32, {order = #NHWC}>
   %vf_0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x128x68x120xf16, {order = #NHWC}>,
                             %cst_4 as %arg4: tensor<64x128x1x1xf16, {order = #NHWC}>,
                             %cst_5 as %arg6: tensor<128x64x3x3xf16, {order = #NHWC}>,
                             %cst_3 as %arg8: tensor<1x128x139x243xi1, {order = #NHWC}>,
                             %storage_0 as %arg9: tensor<1x1x139x243xi32, {order = #NHWC}>,
                             %cst_6 as %arg10: tensor<64x128x4x4xf16, {order = #NHWC}>,
                             %arg2 as %arg12: tensor<1x64x136x240xf16, {order = #NHWC}>,
                             %cst_7 as %arg13: tensor<32x64x1x1xf16, {order = #NHWC}>,
                             %cst_8 as %arg15: tensor<64x32x3x3xf16, {order = #NHWC}>,
                             %cst_1 as %arg17: tensor<1x64x275x483xi1, {order = #NHWC}>,
                             %storage_1 as %arg18: tensor<1x1x275x483xi32, {order = #NHWC}>,
                             %cst_9 as %arg19: tensor<32x64x4x4xf16, {order = #NHWC}>,
                             %arg1 as %arg21: tensor<1x32x272x480xf16, {order = #NHWC}>,
                             %cst_10 as %arg22: tensor<32x32x3x3xf16, {order = #NHWC}>
                             ) attributes {tilingStrategy = [1, 1, 1, 5]} -> tensor<1x32x272x480xf16, {order = #NHWC}> {
      %0 = VPU.NCE.Convolution(%arg3, %arg4) rawFilterShape [64, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x128x68x120xf16, {order = #NHWC}>, tensor<64x128x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      %1 = VPU.NCE.Convolution(%0, %arg6) rawFilterShape [128, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<128x64x3x3xf16, {order = #NHWC}> -> tensor<1x128x68x120xf16, {order = #NHWC}>
      %2 = VPU.NCE.Eltwise(%1, %arg3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x128x68x120xf16, {order = #NHWC}>
      %3 = VPU.GroupSparseTensor(%2, %arg8, %arg9) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>} -> !VPU.SparseTensor<data=tensor<1x128x68x120xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x139x243xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x139x243xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>
      %4 = VPU.NCE.Convolution(%3, %arg10) rawFilterShape [64, 128, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x128x68x120xf16, {order = #NHWC}>, sparsity_map=tensor<1x128x139x243xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x139x243xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>, tensor<64x128x4x4xf16, {order = #NHWC}> -> tensor<1x64x136x240xf16, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%4, %arg12) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x64x136x240xf16, {order = #NHWC}>
      %6 = VPU.NCE.Convolution(%5, %arg13) rawFilterShape [32, 64, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x64x136x240xf16, {order = #NHWC}>, tensor<32x64x1x1xf16, {order = #NHWC}> -> tensor<1x32x136x240xf16, {order = #NHWC}>
      %7 = VPU.NCE.Convolution(%6, %arg15) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x136x240xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}> -> tensor<1x64x136x240xf16, {order = #NHWC}>
      %8 = VPU.NCE.Eltwise(%7, %5) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x64x136x240xf16, {order = #NHWC}>
      %9 = VPU.GroupSparseTensor(%8, %arg17, %arg18) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>} -> !VPU.SparseTensor<data=tensor<1x64x136x240xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x275x483xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x275x483xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>
      %10 = VPU.NCE.Convolution(%9, %arg19) rawFilterShape [32, 64, 4, 4] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x64x136x240xf16, {order = #NHWC}>, sparsity_map=tensor<1x64x275x483xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x275x483xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [2, 2, 2, 2]>>, tensor<32x64x4x4xf16, {order = #NHWC}> -> tensor<1x32x272x480xf16, {order = #NHWC}>
      %11 = VPU.NCE.Eltwise(%10, %arg21) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x272x480xf16, {order = #NHWC}>
      %12 = VPU.NCE.Convolution(%11, %arg22) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>,  strides = [1, 1]} : tensor<1x32x272x480xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x272x480xf16, {order = #NHWC}>
      VPU.Yield %12
    }

    return %vf_0 : tensor<1x32x272x480xf16, {order = #NHWC}>


    // CHECK: [[SLICE_ARG0_0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 128, 68, 27]
    // CHECK: VPU.NCE.Convolution
    // CHECK: [[CONV_1:%.+]] = VPU.NCE.Convolution
    // CHECK: [[SLICE_0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 128, 68, 26]
    // CHECK: [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[SLICE_0]])
    // CHECK: [[GROUP_SPARSE_0:%.+]] = VPU.GroupSparseTensor([[ELTWISE_0]]
    // CHECK: [[CONV_2:%.+]] = VPU.NCE.Convolution([[GROUP_SPARSE_0]]
    // CHECK: [[SLICE_ARG2:%.+]] = VPU.Slice [[ARG_2]] [0, 0, 0, 0] [1, 64, 136, 50]
    // CHECK: [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[CONV_2]], [[SLICE_ARG2]])
    // CHECK: [[CONV_3:%.+]] = VPU.NCE.Convolution([[ELTWISE_1]]
    // CHECK: [[CONV_4:%.+]] = VPU.NCE.Convolution([[CONV_3]]
    // CHECK: [[SLICE_1:%.+]] = VPU.Slice [[ELTWISE_1]] [0, 0, 0, 0] [1, 64, 136, 49]
    // CHECK: [[ELTWISE_2:%.+]] = VPU.NCE.Eltwise([[CONV_4]], [[SLICE_1]])
    // CHECK: [[GROUP_SPARSE_1:%.+]] = VPU.GroupSparseTensor([[ELTWISE_2]]
    // CHECK: [[CONV_5:%.+]] = VPU.NCE.Convolution([[GROUP_SPARSE_1]]
    // CHECK: [[SLICE_ARG1:%.+]] = VPU.Slice [[ARG_1]] [0, 0, 0, 0] [1, 32, 272, 97]
    // CHECK: [[ELTWISE_3:%.+]] = VPU.NCE.Eltwise([[CONV_5]], [[SLICE_ARG1]])
    // CHECK: VPU.NCE.Convolution([[ELTWISE_3]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @TileTwo1x1ConvInputsEltwiseWithoutSliceInserted
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x16x256x256x!qElemType, {order = #NHWC}>
func.func @TileTwo1x1ConvInputsEltwiseWithoutSliceInserted(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {

   %cst_4 = const.Declare tensor<16x32x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<16x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<32x32x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x32x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_6 = const.Declare tensor<32x16x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_6 as %arg2: tensor<32x16x1x1x!qElemType, {order = #NHWC}>, %cst_5 as %arg4: tensor<32x32x1x1x!qElemType, {order = #NHWC}>, %cst_4 as %arg6: tensor<16x32x3x3x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 6, 1]} -> tensor<1x16x256x256x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x1x1x!qElemType, {order = #NHWC}> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %arg6) rawFilterShape [16, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<16x32x3x3x!qElemType, {order = #NHWC}> -> tensor<1x16x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %4
    }

    return %0 : tensor<1x16x256x256x!qElemType, {order = #NHWC}>


    // CHECK:    [[SLICE_ARG_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 44, 256]
    // CHECK:    [[CONV0_TILE0:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_0]]
    // CHECK:    [[CONV1_TILE0:%.+]] = VPU.NCE.Convolution([[CONV0_TILE0]]
    // CHECK:    [[ELTWISE_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE0]], [[CONV1_TILE0]])
    // CHECK:    [[CONV2_TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE0]]

    // CHECK:    [[SLICE_ARG_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 16, 45, 256]
    // CHECK:    [[CONV0_TILE1:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_1]]
    // CHECK:    [[CONV1_TILE1:%.+]] = VPU.NCE.Convolution([[CONV0_TILE1]]
    // CHECK:    [[ELTWISE_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE1]], [[CONV1_TILE1]])
    // CHECK:    [[CONV2_TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE1]]

    // CHECK:    [[SLICE_ARG_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 85, 0] [1, 16, 45, 256]
    // CHECK:    [[CONV0_TILE2:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_2]]
    // CHECK:    [[CONV1_TILE2:%.+]] = VPU.NCE.Convolution([[CONV0_TILE2]]
    // CHECK:    [[ELTWISE_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE2]], [[CONV1_TILE2]]
    // CHECK:    [[CONV2_TILE2:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE2]]

    // CHECK:    [[SLICE_ARG_3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 16, 45, 256]
    // CHECK:    [[CONV0_TILE3:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_3]]
    // CHECK:    [[CONV1_TILE3:%.+]] = VPU.NCE.Convolution([[CONV0_TILE3]]
    // CHECK:    [[ELTWISE_TILE3:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE3]], [[CONV1_TILE3]])
    // CHECK:    [[CONV2_TILE3:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE3]]

    // CHECK:    [[SLICE_ARG_4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 171, 0] [1, 16, 44, 256]
    // CHECK:    [[CONV0_TILE4:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_4]]
    // CHECK:    [[CONV1_TILE4:%.+]] = VPU.NCE.Convolution([[CONV0_TILE4]]
    // CHECK:    [[ELTWISE_TILE4:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE4]], [[CONV1_TILE4]]
    // CHECK:    [[CONV2_TILE4:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE4]]

    // CHECK:    [[SLICE_ARG_5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 213, 0] [1, 16, 43, 256]
    // CHECK:    [[CONV0_TILE5:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_5]]
    // CHECK:    [[CONV1_TILE5:%.+]] = VPU.NCE.Convolution([[CONV0_TILE5]]
    // CHECK:    [[ELTWISE_TILE5:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE5]], [[CONV1_TILE5]]
    // CHECK:    [[CONV2_TILE5:%.+]] = VPU.NCE.Convolution([[ELTWISE_TILE5]]

    // CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CONV2_TILE0]], [[CONV2_TILE1]], [[CONV2_TILE2]], [[CONV2_TILE3]], [[CONV2_TILE4]], [[CONV2_TILE5]]
    // CHECK:    return [[CONCAT]] : tensor<1x16x256x256x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @TileTwo1x1ConvInputsEltwiseWithSliceInserted
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x384x65x65x!qElemType, {order = #NHWC}>
func.func @TileTwo1x1ConvInputsEltwiseWithSliceInserted(%arg0: tensor<1x384x65x65x!qElemType, {order = #NHWC}>) -> tensor<1x96x65x65x!qElemType, {order = #NHWC}> {
   %cst_1 = const.Declare tensor<96x384x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<96x384x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_3 = const.Declare tensor<576x96x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<576x96x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<576x32x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<576x32x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

   %cst_7 = const.Declare tensor<96x576x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<96x576x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x384x65x65x!qElemType, {order = #NHWC}>,
                            %cst_1 as %arg2: tensor<96x384x1x1x!qElemType, {order = #NHWC}>,
                            %cst_3 as %arg4: tensor<576x96x1x1x!qElemType, {order = #NHWC}>,
                            %cst_5 as %arg6: tensor<576x32x1x1x!qElemType, {order = #NHWC}>,
                            %cst_7 as %arg7: tensor<96x576x1x1x!qElemType, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x96x65x65x!qElemType, {order = #NHWC}> {

      %1 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [96, 384, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x384x65x65x!qElemType, {order = #NHWC}>, tensor<96x384x1x1x!qElemType, {order = #NHWC}> -> tensor<1x96x65x65x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4) rawFilterShape [576, 96, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x96x65x65x!qElemType, {order = #NHWC}>, tensor<576x96x1x1x!qElemType, {order = #NHWC}> -> tensor<1x576x65x65x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.DepthConvolution(%2, %arg6) rawFilterShape [576, 1, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} -> tensor<1x576x65x65x!qElemType, {order = #NHWC}>
      %4 = VPU.NCE.Convolution(%3, %arg7) rawFilterShape [96, 576, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x576x65x65x!qElemType, {order = #NHWC}>, tensor<96x576x1x1x!qElemType, {order = #NHWC}> -> tensor<1x96x65x65x!qElemType, {order = #NHWC}>
      %5 = VPU.NCE.Eltwise(%1, %4)
             {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              op_type = #VPU.eltwise_type<ADD>,
              ppe = #VPU.PPEStub<>} -> tensor<1x96x65x65x!qElemType, {order = #NHWC}>
      VPU.Yield %5
  }
  return %0 : tensor<1x96x65x65x!qElemType, {order = #NHWC}>

  // CHECK:     [[SLICE_ARG_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 384, 24, 65]
  // CHECK:     [[CONV0_TILE0:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_0]]
  // CHECK:     [[CONV1_TILE0:%.+]] = VPU.NCE.Convolution([[CONV0_TILE0]]
  // CHECK:     [[CONV2_TILE0:%.+]] = VPU.NCE.DepthConvolution([[CONV1_TILE0]]
  // CHECK:     [[CONV3_TILE0:%.+]] = VPU.NCE.Convolution([[CONV2_TILE0]]
  // CHECK:     [[CONV0_SLICE_TILE0:%.+]] = VPU.Slice [[CONV0_TILE0]] [0, 0, 0, 0] [1, 96, 22, 65]
  // CHECK:     [[ELTWISE_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE0]], [[CONV3_TILE0]])

  // CHECK:     [[SLICE_ARG_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 20, 0] [1, 384, 26, 65]
  // CHECK:     [[CONV0_TILE1:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_1]]
  // CHECK:     [[CONV1_TILE1:%.+]] = VPU.NCE.Convolution([[CONV0_TILE1]]
  // CHECK:     [[CONV2_TILE1:%.+]] = VPU.NCE.DepthConvolution([[CONV1_TILE1]]
  // CHECK:     [[CONV3_TILE1:%.+]] = VPU.NCE.Convolution([[CONV2_TILE1]]
  // CHECK:     [[CONV0_SLICE_TILE1:%.+]] = VPU.Slice [[CONV0_TILE1]] [0, 0, 2, 0] [1, 96, 22, 65]
  // CHECK:     [[ELTWISE_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE1]], [[CONV3_TILE1]])

  // CHECK:     [[SLICE_ARG_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 384, 23, 65]
  // CHECK:     [[CONV0_TILE2:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_2]]
  // CHECK:     [[CONV1_TILE2:%.+]] = VPU.NCE.Convolution([[CONV0_TILE2]]
  // CHECK:     [[CONV2_TILE2:%.+]] = VPU.NCE.DepthConvolution([[CONV1_TILE2]]
  // CHECK:     [[CONV3_TILE2:%.+]] = VPU.NCE.Convolution([[CONV2_TILE2]]
  // CHECK:     [[CONV0_SLICE_TILE2:%.+]] = VPU.Slice [[CONV0_TILE2]] [0, 0, 2, 0] [1, 96, 21, 65]
  // CHECK:     [[ELTWISE_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE2]], [[CONV3_TILE2]])
  // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_TILE0]], [[ELTWISE_TILE1]], [[ELTWISE_TILE2]])
  // CHECK:     return [[CONCAT]] : tensor<1x96x65x65x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @TileTwo1x1ConvInputsEltwiseAnd5x5ConvVFInputWithSliceInserted
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x384x65x65x!qElemType, {order = #NHWC}>
func.func @TileTwo1x1ConvInputsEltwiseAnd5x5ConvVFInputWithSliceInserted(%arg0: tensor<1x384x65x65x!qElemType, {order = #NHWC}>) -> tensor<1x96x65x65x!qElemType, {order = #NHWC}> {
   %cst_1 = const.Declare tensor<384x32x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<384x32x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

   %cst_3 = const.Declare tensor<96x384x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<96x384x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_5 = const.Declare tensor<576x96x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<576x96x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_7 = const.Declare tensor<576x32x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<576x32x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

   %cst_9 = const.Declare tensor<96x576x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<96x576x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x384x65x65x!qElemType, {order = #NHWC}>,
                            %cst_1 as %arg2: tensor<384x32x1x1x!qElemType, {order = #NHWC}>,
                            %cst_3 as %arg3: tensor<96x384x1x1x!qElemType, {order = #NHWC}>,
                            %cst_5 as %arg5: tensor<576x96x1x1x!qElemType, {order = #NHWC}>,
                            %cst_7 as %arg7: tensor<576x32x1x1x!qElemType, {order = #NHWC}>,
                            %cst_9 as %arg8: tensor<96x576x1x1x!qElemType, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x96x65x65x!qElemType, {order = #NHWC}> {

      %1 = VPU.NCE.DepthConvolution(%arg1, %arg2) rawFilterShape [384, 1, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} -> tensor<1x384x65x65x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg3) rawFilterShape [96, 384, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x384x65x65x!qElemType, {order = #NHWC}>, tensor<96x384x1x1x!qElemType, {order = #NHWC}> -> tensor<1x96x65x65x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Convolution(%2, %arg5) rawFilterShape [576, 96, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x96x65x65x!qElemType, {order = #NHWC}>, tensor<576x96x1x1x!qElemType, {order = #NHWC}> -> tensor<1x576x65x65x!qElemType, {order = #NHWC}>
      %4 = VPU.NCE.DepthConvolution(%3, %arg7) rawFilterShape [576, 1, 5, 5] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 2 : i64, right = 2 : i64, top = 2 : i64, bottom = 2 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} -> tensor<1x576x65x65x!qElemType, {order = #NHWC}>
      %5 = VPU.NCE.Convolution(%4, %arg8) rawFilterShape [96, 576, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x576x65x65x!qElemType, {order = #NHWC}>, tensor<96x576x1x1x!qElemType, {order = #NHWC}> -> tensor<1x96x65x65x!qElemType, {order = #NHWC}>
      %6 = VPU.NCE.Eltwise(%2, %5)
             {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
              op_type = #VPU.eltwise_type<ADD>,
              ppe = #VPU.PPEStub<>} -> tensor<1x96x65x65x!qElemType, {order = #NHWC}>
      VPU.Yield %6
  }
  return %0 : tensor<1x96x65x65x!qElemType, {order = #NHWC}>

  // CHECK:     [[SLICE_ARG_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 384, 26, 65]
  // CHECK:     [[CONV0_TILE0:%.+]] = VPU.NCE.DepthConvolution([[SLICE_ARG_0]]
  // CHECK:     [[CONV1_TILE0:%.+]] = VPU.NCE.Convolution([[CONV0_TILE0]]
  // CHECK:     [[CONV2_TILE0:%.+]] = VPU.NCE.Convolution([[CONV1_TILE0]]
  // CHECK:     [[CONV3_TILE0:%.+]] = VPU.NCE.DepthConvolution([[CONV2_TILE0]]
  // CHECK:     [[CONV4_TILE0:%.+]] = VPU.NCE.Convolution([[CONV3_TILE0]]
  // CHECK:     [[CONV1_SLICE_TILE0:%.+]] = VPU.Slice [[CONV1_TILE0]] [0, 0, 0, 0] [1, 96, 22, 65]
  // CHECK:     [[ELTWISE_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV1_SLICE_TILE0]], [[CONV4_TILE0]])

  // CHECK:     [[SLICE_ARG_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 18, 0] [1, 384, 30, 65]
  // CHECK:     [[CONV0_TILE1:%.+]] = VPU.NCE.DepthConvolution([[SLICE_ARG_1]]
  // CHECK:     [[CONV1_TILE1:%.+]] = VPU.NCE.Convolution([[CONV0_TILE1]]
  // CHECK:     [[CONV2_TILE1:%.+]] = VPU.NCE.Convolution([[CONV1_TILE1]]
  // CHECK:     [[CONV3_TILE1:%.+]] = VPU.NCE.DepthConvolution([[CONV2_TILE1]]
  // CHECK:     [[CONV4_TILE1:%.+]] = VPU.NCE.Convolution([[CONV3_TILE1]]
  // CHECK:     [[CONV1_SLICE_TILE1:%.+]] = VPU.Slice [[CONV1_TILE1]] [0, 0, 2, 0] [1, 96, 22, 65]
  // CHECK:     [[ELTWISE_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV1_SLICE_TILE1]], [[CONV4_TILE1]])

  // CHECK:     [[SLICE_ARG_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 384, 25, 65]
  // CHECK:     [[CONV0_TILE2:%.+]] = VPU.NCE.DepthConvolution([[SLICE_ARG_2]]
  // CHECK:     [[CONV1_TILE2:%.+]] = VPU.NCE.Convolution([[CONV0_TILE2]]
  // CHECK:     [[CONV2_TILE2:%.+]] = VPU.NCE.Convolution([[CONV1_TILE2]]
  // CHECK:     [[CONV3_TILE2:%.+]] = VPU.NCE.DepthConvolution([[CONV2_TILE2]]
  // CHECK:     [[CONV4_TILE2:%.+]] = VPU.NCE.Convolution([[CONV3_TILE2]]
  // CHECK:     [[CONV1_SLICE_TILE2:%.+]] = VPU.Slice [[CONV1_TILE2]] [0, 0, 2, 0] [1, 96, 21, 65]
  // CHECK:     [[ELTWISE_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV1_SLICE_TILE2]], [[CONV4_TILE2]])
  // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_TILE0]], [[ELTWISE_TILE1]], [[ELTWISE_TILE2]])
  // CHECK:     return [[CONCAT]] : tensor<1x96x65x65x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>


!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @TileDifferentInputsEltwise
// CHECK-SAME:   [[INPUT:%.+]]: tensor<1x16x128x128x!qElemType, {order = #NHWC}>
func.func @TileDifferentInputsEltwise(%arg0: tensor<1x16x128x128x!qElemType, {order = #NHWC}>) -> tensor<1x32x128x128x!qElemType, {order = #NHWC}> {

   %cst_0 = const.Declare tensor<32x16x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x16x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %cst_2 = const.Declare tensor<32x32x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
   %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x128x128x!qElemType, {order = #NHWC}>,
                            %cst_0 as %arg2: tensor<32x16x1x1x!qElemType, {order = #NHWC}>,
                            %cst_2 as %arg4: tensor<32x32x3x3x!qElemType, {order = #NHWC}>
                            ) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x32x128x128x!qElemType, {order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2) rawFilterShape [32, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x16x128x128x!qElemType, {order = #NHWC}>, tensor<32x16x1x1x!qElemType, {order = #NHWC}> -> tensor<1x32x128x128x!qElemType, {order = #NHWC}>
      %2 = VPU.NCE.Convolution(%1, %arg4) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
         ppe = #VPU.PPEStub<>,
          strides = [1, 1]}
         : tensor<1x32x128x128x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x128x128x!qElemType, {order = #NHWC}>
      %3 = VPU.NCE.Eltwise(%1, %2)
         {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x128x128x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
  return %0 : tensor<1x32x128x128x!qElemType, {order = #NHWC}>

  // CHECK:     [[SLICE_ARG_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 44, 128]
  // CHECK:     [[CONV0_TILE0:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_0]]
  // CHECK:     [[CONV1_TILE0:%.+]] = VPU.NCE.Convolution([[CONV0_TILE0]]
  // CHECK:     [[CONV0_SLICE_TILE0:%.+]] = VPU.Slice [[CONV0_TILE0]] [0, 0, 0, 0] [1, 32, 43, 128]
  // CHECK:     [[ELTWISE_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE0]], [[CONV1_TILE0]])

  // CHECK:     [[SLICE_ARG_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 16, 45, 128]
  // CHECK:     [[CONV0_TILE1:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_1]]
  // CHECK:     [[CONV1_TILE1:%.+]] = VPU.NCE.Convolution([[CONV0_TILE1]]
  // CHECK:     [[CONV0_SLICE_TILE1:%.+]] = VPU.Slice [[CONV0_TILE1]] [0, 0, 1, 0] [1, 32, 43, 128]
  // CHECK:     [[ELTWISE_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE1]], [[CONV1_TILE1]])

  // CHECK:     [[SLICE_ARG_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 85, 0] [1, 16, 43, 128]
  // CHECK:     [[CONV0_TILE2:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_2]]
  // CHECK:     [[CONV1_TILE2:%.+]] = VPU.NCE.Convolution([[CONV0_TILE2]]
  // CHECK:     [[CONV0_SLICE_TILE2:%.+]] = VPU.Slice [[CONV0_TILE2]] [0, 0, 1, 0] [1, 32, 42, 128]
  // CHECK:     [[ELTWISE_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE2]], [[CONV1_TILE2]])
  // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_TILE0]], [[ELTWISE_TILE1]], [[ELTWISE_TILE2]])
  // CHECK:     return [[CONCAT]] : tensor<1x32x128x128x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.022970187430288277:92>

// CHECK-LABEL: func.func @TileTwoEltwiseInParallelWithSameInputs
// CHECK-SAME:   [[INPUT0:%.+]]: tensor<1x32x72x128x!qElemType, {order = #NHWC}>
// CHECK-SAME:   [[INPUT1:%.+]]: tensor<1x64x72x128x!qElemType, {order = #NHWC}>
// CHECK-SAME:   [[INPUT2:%.+]]: tensor<1x128x72x128x!qElemType, {order = #NHWC}>
func.func @TileTwoEltwiseInParallelWithSameInputs(%arg0: tensor<1x32x72x128x!qElemType, {order = #NHWC}>,
                                                  %arg1: tensor<1x64x72x128x!qElemType, {order = #NHWC}>,
                                                  %arg2: tensor<1x128x72x128x!qElemType, {order = #NHWC}>)
                                                  -> tensor<1x64x72x128x!qElemType, {order = #NHWC}> {
  %cst_1 = const.Declare tensor<64x128x1x1x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<64x128x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %cst_3 = const.Declare tensor<64x32x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  %cst_4 = const.Declare tensor<64x64x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

  %0 = VPU.VerticalFusion (%arg2 as %arg3: tensor<1x128x72x128x!qElemType, {order = #NHWC}>,
                           %cst_1 as %arg4: tensor<64x128x1x1x!qElemType, {order = #NHWC}>,
                           %arg0 as %arg6: tensor<1x32x72x128x!qElemType, {order = #NHWC}>,
                           %cst_3 as %arg7: tensor<64x32x3x3x!qElemType, {order = #NHWC}>,
                           %cst_4 as %arg9: tensor<64x64x3x3x!qElemType, {order = #NHWC}>,
                           %cst_4 as %arg11: tensor<64x64x3x3x!qElemType,{order = #NHWC}>,
                           %arg1 as %arg13: tensor<1x64x72x128x!qElemType, {order = #NHWC}>,
                           %cst_4 as %arg14: tensor<64x64x3x3x!qElemType, {order = #NHWC}>,
                           %cst_4 as %arg16: tensor<64x64x3x3x!qElemType, {order = #NHWC}>
                           ) attributes {scenario = #VPU.vf_scenario<FULL_PREFETCHING>, tilingStrategy = [1, 1, 1, 4]}
                           -> tensor<1x64x72x128x!qElemType, {order = #NHWC}> {

      %1 = VPU.NCE.Convolution(%arg3, %arg4) rawFilterShape [64, 128, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x128x72x128x!qElemType, {order = #NHWC}>, tensor<64x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %2 = VPU.NCE.Convolution(%arg6, %arg7) rawFilterShape [64, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x32x72x128x!qElemType, {order = #NHWC}>, tensor<64x32x3x3x!qElemType, {order = #NHWC}> -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %3 = VPU.NCE.Convolution(%2, %arg9) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x64x72x128x!qElemType, {order = #NHWC}>, tensor<64x64x3x3x!qElemType, {order = #NHWC}> -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %4 = VPU.NCE.Convolution(%3, %arg11) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x64x72x128x!qElemType, {order = #NHWC}>, tensor<64x64x3x3x!qElemType,{order = #NHWC}> -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %5 = VPU.NCE.Eltwise(%4, %arg13)
             {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>,
              ppe = #VPU.PPEStub<>} -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %6 = VPU.NCE.Eltwise(%1, %5)
             {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>,
              ppe = #VPU.PPEStub<>} -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %7 = VPU.NCE.Eltwise(%1, %5)
             {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>,
              ppe = #VPU.PPEStub<>} -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %8 = VPU.NCE.Convolution(%7, %arg14) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x64x72x128x!qElemType, {order = #NHWC}>, tensor<64x64x3x3x!qElemType, {order = #NHWC}> -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %9 = VPU.NCE.Convolution(%8, %arg16) rawFilterShape [64, 64, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
              ppe = #VPU.PPEStub<>,
               strides = [1, 1]} : tensor<1x64x72x128x!qElemType, {order = #NHWC}>, tensor<64x64x3x3x!qElemType, {order = #NHWC}> -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      %10 = VPU.NCE.Eltwise(%9, %6)
             {resultSegmentSizes = array<i32: 1, 0, 0, 0>, is_inplace = true, op_type = #VPU.eltwise_type<ADD>,
              ppe = #VPU.PPEStub<>} -> tensor<1x64x72x128x!qElemType, {order = #NHWC}>

      VPU.Yield %10
  }

  return %0 : tensor<1x64x72x128x!qElemType, {order = #NHWC}>

  // CHECK:     [[SLICE_ARG_0_0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 128, 72, 34]
  // CHECK:     [[CONV0_TILE0:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_0_0]]
  // CHECK:     [[SLICE_ARG_0_1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 32, 72, 37]
  // CHECK:     [[CONV1_TILE0:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_0_1]]
  // CHECK:     [[CONV2_TILE0:%.+]] = VPU.NCE.Convolution([[CONV1_TILE0]]
  // CHECK:     [[CONV3_TILE0:%.+]] = VPU.NCE.Convolution([[CONV2_TILE0]]
  // CHECK:     [[SLICE_ARG_0_2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 64, 72, 34]
  // CHECK:     [[ELTWISE0_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV3_TILE0]], [[SLICE_ARG_0_2]])
  // CHECK:     [[CONV0_SLICE_TILE0:%.+]] = VPU.Slice [[CONV0_TILE0]] [0, 0, 0, 0] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE0_SLICE_TILE0:%.+]] = VPU.Slice [[ELTWISE0_TILE0]] [0, 0, 0, 0] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE1_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE0]], [[ELTWISE0_SLICE_TILE0]])
  // CHECK:     [[ELTWISE2_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE0]], [[ELTWISE0_TILE0]])
  // CHECK:     [[CONV4_TILE0:%.+]] = VPU.NCE.Convolution([[ELTWISE2_TILE0]]
  // CHECK:     [[CONV5_TILE0:%.+]] = VPU.NCE.Convolution([[CONV4_TILE0]]
  // CHECK:     [[ELTWISE3_TILE0:%.+]] = VPU.NCE.Eltwise([[CONV5_TILE0]], [[ELTWISE1_TILE0]])

  // CHECK:     [[SLICE_ARG_1_0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 30] [1, 128, 72, 36]
  // CHECK:     [[CONV0_TILE1:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_1_0]]
  // CHECK:     [[SLICE_ARG_1_1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 27] [1, 32, 72, 42]
  // CHECK:     [[CONV1_TILE1:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_1_1]]
  // CHECK:     [[CONV2_TILE1:%.+]] = VPU.NCE.Convolution([[CONV1_TILE1]]
  // CHECK:     [[CONV3_TILE1:%.+]] = VPU.NCE.Convolution([[CONV2_TILE1]]
  // CHECK:     [[SLICE_ARG_1_2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 30] [1, 64, 72, 36]
  // CHECK:     [[ELTWISE0_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV3_TILE1]], [[SLICE_ARG_1_2]])
  // CHECK:     [[CONV0_SLICE_TILE1:%.+]] = VPU.Slice [[CONV0_TILE1]] [0, 0, 0, 2] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE0_SLICE_TILE1:%.+]] = VPU.Slice [[ELTWISE0_TILE1]] [0, 0, 0, 2] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE1_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE1]], [[ELTWISE0_SLICE_TILE1]])
  // CHECK:     [[ELTWISE2_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE1]], [[ELTWISE0_TILE1]])
  // CHECK:     [[CONV4_TILE1:%.+]] = VPU.NCE.Convolution([[ELTWISE2_TILE1]]
  // CHECK:     [[CONV5_TILE1:%.+]] = VPU.NCE.Convolution([[CONV4_TILE1]]
  // CHECK:     [[ELTWISE3_TILE1:%.+]] = VPU.NCE.Eltwise([[CONV5_TILE1]], [[ELTWISE1_TILE1]])

  // CHECK:     [[SLICE_ARG_2_0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 62] [1, 128, 72, 36]
  // CHECK:     [[CONV0_TILE2:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_2_0]]
  // CHECK:     [[SLICE_ARG_2_1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 59] [1, 32, 72, 42]
  // CHECK:     [[CONV1_TILE2:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_2_1]]
  // CHECK:     [[CONV2_TILE2:%.+]] = VPU.NCE.Convolution([[CONV1_TILE2]]
  // CHECK:     [[CONV3_TILE2:%.+]] = VPU.NCE.Convolution([[CONV2_TILE2]]
  // CHECK:     [[SLICE_ARG_2_2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 62] [1, 64, 72, 36]
  // CHECK:     [[ELTWISE0_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV3_TILE2]], [[SLICE_ARG_2_2]])
  // CHECK:     [[CONV0_SLICE_TILE2:%.+]] = VPU.Slice [[CONV0_TILE2]] [0, 0, 0, 2] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE0_SLICE_TILE2:%.+]] = VPU.Slice [[ELTWISE0_TILE2]] [0, 0, 0, 2] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE1_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE2]], [[ELTWISE0_SLICE_TILE2]])
  // CHECK:     [[ELTWISE2_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE2]], [[ELTWISE0_TILE2]])
  // CHECK:     [[CONV4_TILE2:%.+]] = VPU.NCE.Convolution([[ELTWISE2_TILE2]]
  // CHECK:     [[CONV5_TILE2:%.+]] = VPU.NCE.Convolution([[CONV4_TILE2]]
  // CHECK:     [[ELTWISE3_TILE2:%.+]] = VPU.NCE.Eltwise([[CONV5_TILE2]], [[ELTWISE1_TILE2]])

  // CHECK:     [[SLICE_ARG_3_0:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 94] [1, 128, 72, 34]
  // CHECK:     [[CONV0_TILE3:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_3_0]]
  // CHECK:     [[SLICE_ARG_3_1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 91] [1, 32, 72, 37]
  // CHECK:     [[CONV1_TILE3:%.+]] = VPU.NCE.Convolution([[SLICE_ARG_3_1]]
  // CHECK:     [[CONV2_TILE3:%.+]] = VPU.NCE.Convolution([[CONV1_TILE3]]
  // CHECK:     [[CONV3_TILE3:%.+]] = VPU.NCE.Convolution([[CONV2_TILE3]]
  // CHECK:     [[SLICE_ARG_3_2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 94] [1, 64, 72, 34]
  // CHECK:     [[ELTWISE0_TILE3:%.+]] = VPU.NCE.Eltwise([[CONV3_TILE3]], [[SLICE_ARG_3_2]])
  // CHECK:     [[CONV0_SLICE_TILE3:%.+]] = VPU.Slice [[CONV0_TILE3]] [0, 0, 0, 2] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE0_SLICE_TILE3:%.+]] = VPU.Slice [[ELTWISE0_TILE3]] [0, 0, 0, 2] [1, 64, 72, 32]
  // CHECK:     [[ELTWISE1_TILE3:%.+]] = VPU.NCE.Eltwise([[CONV0_SLICE_TILE3]], [[ELTWISE0_SLICE_TILE3]])
  // CHECK:     [[ELTWISE2_TILE3:%.+]] = VPU.NCE.Eltwise([[CONV0_TILE3]], [[ELTWISE0_TILE3]])
  // CHECK:     [[CONV4_TILE3:%.+]] = VPU.NCE.Convolution([[ELTWISE2_TILE3]]
  // CHECK:     [[CONV5_TILE3:%.+]] = VPU.NCE.Convolution([[CONV4_TILE3]]
  // CHECK:     [[ELTWISE3_TILE3:%.+]] = VPU.NCE.Eltwise([[CONV5_TILE3]], [[ELTWISE1_TILE3]])

  // CHECK:     [[CONCAT:%.+]] = VPU.Concat([[ELTWISE3_TILE0]], [[ELTWISE3_TILE1]], [[ELTWISE3_TILE2]], [[ELTWISE3_TILE3]])

  // CHECK:     return [[CONCAT]] : tensor<1x64x72x128x!qElemType, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VfTilingWithAffineReshapePermuteCastConvEltwise
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x1024x4x1024xf16>,
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<1x4096x1x1xf16, {order = #NHWC}>,
// CHECK-SAME: [[ARG_2:%[^:]+]]: tensor<1x1024x1x1xf16, {order = #NHWC}>)
func.func @VfTilingWithAffineReshapePermuteCastConvEltwise(
    %arg0: tensor<1x1024x4x1024xf16>,
    %arg1: tensor<1x4096x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<1x1024x1x1xf16, {order = #NHWC}>)
    -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

    %cst_wt = const.Declare tensor<1024x1x1x1xf32> = dense<1.0> : tensor<1024x1x1x1xf32>
    %cst_bias = const.Declare tensor<1024x1x1x1xf32> = dense<0.0> : tensor<1024x1x1x1xf32>

    %0 = VPU.VerticalFusion (
        %arg0 as %arg3: tensor<1x1024x4x1024xf16>,
        %arg1 as %arg4: tensor<1x4096x1x1xf16, {order = #NHWC}>,
        %arg2 as %arg5: tensor<1x1024x1x1xf16, {order = #NHWC}>
    ) attributes {scenario = #VPU.vf_scenario<LASTOP_PREFETCHING>, tilingStrategy = [1, 2, 1, 1], vf_loop_index = 0}
      -> tensor<1x1024x1x1xf16, {order = #NHWC}> {

      %1 = VPU.AffineReshape(%arg3) {
          dim_mapping = [[0], [0], [1], [1, 2, 3]],
          shape_value = [1024, 4096, 1, 1]
      } : tensor<1x1024x4x1024xf16> -> tensor<1024x4096x1x1xf16>

      %2 = VPU.PermuteCast(%1) {
          dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
          mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
      } : tensor<1024x4096x1x1xf16>
        -> tensor<1024x4096x1x1xf16, {order = #NHWC}>

      %3 = VPU.NCE.Convolution(%arg4, %2) rawFilterShape [1024, 4096, 1, 1] {
          mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
          ppe = #VPU.PPEStub<>,
          resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          strides = [1, 1]
      } : tensor<1x4096x1x1xf16, {order = #NHWC}>,
          tensor<1024x4096x1x1xf16, {order = #NHWC}>
        -> tensor<1x1024x1x1xf16, {order = #NHWC}>

      %4 = VPU.NCE.Eltwise(%arg5, %3) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
          mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
          multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
          op_type = #VPU.eltwise_type<ADD>,
          ppe = #VPU.PPEStub<>
      } -> tensor<1x1024x1x1xf16, {order = #NHWC}>

      VPU.Yield %4
    }

    return %0 : tensor<1x1024x1x1xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[SLICE_ARG0_0:%.+]] = VPU.Slice [[ARG_0]] [0, 0, 0, 0] [1, 512, 4, 1024]
    // CHECK-SAME:      : tensor<1x1024x4x1024xf16> to tensor<1x512x4x1024xf16>
    // CHECK:       [[RESHAPE_0:%.+]] = VPU.AffineReshape([[SLICE_ARG0_0]])
    // CHECK-SAME:      shape_value = [512, 4096, 1, 1]
    // CHECK-SAME:      : tensor<1x512x4x1024xf16> -> tensor<512x4096x1x1xf16>
    // CHECK:       [[PERMUTE_0:%.+]] = VPU.PermuteCast([[RESHAPE_0]])
    // CHECK-SAME:      : tensor<512x4096x1x1xf16> -> tensor<512x4096x1x1xf16, {order = #NHWC}>

    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[PERMUTE_0]]) rawFilterShape [512, 4096, 1, 1]
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_ARG2_0:%.+]] = VPU.Slice [[ARG_2]] [0, 0, 0, 0] [1, 512, 1, 1]
    // CHECK-SAME:      : tensor<1x1024x1x1xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[SLICE_ARG2_0]], [[CONV_0]])
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x512x1x1xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[SLICE_ARG0_1:%.+]] = VPU.Slice [[ARG_0]] [0, 512, 0, 0] [1, 512, 4, 1024]
    // CHECK-SAME:      : tensor<1x1024x4x1024xf16> to tensor<1x512x4x1024xf16>
    // CHECK:       [[RESHAPE_1:%.+]] = VPU.AffineReshape([[SLICE_ARG0_1]])
    // CHECK-SAME:      shape_value = [512, 4096, 1, 1]
    // CHECK-SAME:      : tensor<1x512x4x1024xf16> -> tensor<512x4096x1x1xf16>
    // CHECK:       [[PERMUTE_1:%.+]] = VPU.PermuteCast([[RESHAPE_1]])
    // CHECK-SAME:      : tensor<512x4096x1x1xf16> -> tensor<512x4096x1x1xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[ARG_1]], [[PERMUTE_1]]) rawFilterShape [512, 4096, 1, 1]
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[SLICE_ARG2_1:%.+]] = VPU.Slice [[ARG_2]] [0, 512, 0, 0] [1, 512, 1, 1]
    // CHECK-SAME:      : tensor<1x1024x1x1xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[SLICE_ARG2_1]], [[CONV_1]])
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x512x1x1xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
    // CHECK-SAME:      {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 512, 0, 0]]}
    // CHECK-SAME:      tensor<1x512x1x1xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x1024x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x1024x1x1xf16, {order = #NHWC}>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1, d3)>


// CHECK-LABEL: @VFTilingWithSDPAMatmul
func.func @VFTilingWithSDPAMatmul(%arg0: tensor<1x32x1x96xf16>, %arg1: tensor<32x1152x96x1x1xf16, {order = #GNHWC}>, %arg2: tensor<1x1152x32x1xf16, {order = #NHWC}>, %arg3: tensor<32x96x1152x1x1xf16, {order = #GNHWC}>, %arg4: tensor<1x1152x1x1x1xf32>, %arg5: tensor<1x1152x1x1x1xf32>, %arg6: tensor<1x96x1x1x1xf32>, %arg7: tensor<1x96x1x1x1xf32>) -> tensor<32x1x96x1x1xf16, {order = #GNHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg8: tensor<1x32x1x96xf16>, %arg1 as %arg9: tensor<32x1152x96x1x1xf16, {order = #GNHWC}>, %arg2 as %arg10: tensor<1x1152x32x1xf16, {order = #NHWC}>, %arg3 as %arg11: tensor<32x96x1152x1x1xf16, {order = #GNHWC}>) attributes {scenario = #VPU.vf_scenario<VF_PIPELINING>, tilingStrategy = [4, 1, 1, 1, 1]} -> tensor<32x1x96x1x1xf16, {order = #GNHWC}> {
      %1 = VPU.AffineReshape(%arg8) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [32, 1, 96, 1, 1]} : tensor<1x32x1x96xf16> -> tensor<32x1x96x1x1xf16>
      %2 = VPU.PermuteCast(%1) {dst_order = #GNHWC, mem_perm = #map} : tensor<32x1x96x1x1xf16> -> tensor<32x1x96x1x1xf16, {order = #GNHWC}>
      %output = VPU.NCE.MatMul(%2, %arg9) rawFilterShape [32, 1152, 96, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<32x1x1152x1x1xf16, {order = #GNHWC}>
      %3 = VPU.PermuteCast(%output) {dst_order = #NCDHW, mem_perm = #map1} : tensor<32x1x1152x1x1xf16, {order = #GNHWC}> -> tensor<32x1x1152x1x1xf16>
      %4 = VPU.AffineReshape(%3) {dim_mapping = [[0, 1], [2], [3], [3], [3]], shape_value = [1, 32, 1, 1152]} : tensor<32x1x1152x1x1xf16> -> tensor<1x32x1x1152xf16>
      %5 = VPU.PermuteCast(%4) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x32x1x1152xf16> -> tensor<1x1152x32x1xf16, {order = #NHWC}>
      %output_0 = VPU.NCE.Eltwise(%5, %arg10) {mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>} -> tensor<1x1152x32x1xf16, {order = #NHWC}>
      %6 = VPU.SoftMax(%output_0) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x1152x32x1xf16, {order = #NHWC}> -> tensor<1x1152x32x1xf16, {order = #NHWC}>
      %7 = VPU.PermuteCast(%6) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x1152x32x1xf16, {order = #NHWC}> -> tensor<1x32x1x1152xf16>
      %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [32, 1, 1152, 1, 1]} : tensor<1x32x1x1152xf16> -> tensor<32x1x1152x1x1xf16>
      %9 = VPU.PermuteCast(%8) {dst_order = #GNHWC, mem_perm = #map} : tensor<32x1x1152x1x1xf16> -> tensor<32x1x1152x1x1xf16, {order = #GNHWC}>
      %output_1 = VPU.NCE.MatMul(%9, %arg11) rawFilterShape [32, 96, 1152, 1, 1] {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, resultSegmentSizes = array<i32: 1, 0, 0, 0>, strides = [1, 1]} -> tensor<32x1x96x1x1xf16, {order = #GNHWC}>
      VPU.Yield %output_1
    }
    return %0 : tensor<32x1x96x1x1xf16, {order = #GNHWC}>
      // CHECK-NOT: VPU.VerticalFusion
      // CHECK: [[ARG0_0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 8, 1, 96]
      // CHECK: [[RESHAPE_0:%.+]] = VPU.AffineReshape([[ARG0_0]])
      // CHECK-SAME: shape_value = [8, 1, 96, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 0 : i64
      // CHECK: [[PERMUTE_0:%.+]] = VPU.PermuteCast([[RESHAPE_0]])
      // CHECK-SAME: vf_loop_tile_index = 0 : i64
      // CHECK: [[FILTER0_0:%.+]] = VPU.Slice %arg1 [0, 0, 0, 0, 0] [8, 1152, 96, 1, 1]
      // CHECK: [[MATMUL0:%.+]] = VPU.NCE.MatMul([[PERMUTE_0]], [[FILTER0_0]]) rawFilterShape [8, 1152, 96, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 0 : i64
      // CHECK-SAME: -> tensor<8x1x1152x1x1xf16, {order = #GNHWC}>
      // CHECK: [[VIEW0:%.+]] = VPU.PermuteCast([[MATMUL0]])
      // CHECK: [[RESHAPE_VIEW0:%.+]] = VPU.AffineReshape([[VIEW0]])
      // CHECK-SAME: shape_value = [1, 8, 1, 1152]
      // CHECK-SAME: vf_loop_tile_index = 0 : i64
      // CHECK: [[QUERY0:%.+]] = VPU.PermuteCast([[RESHAPE_VIEW0]])
      // CHECK: [[ARG2_0:%.+]] = VPU.Slice %arg2 [0, 0, 0, 0] [1, 1152, 8, 1]
      // CHECK: [[ELTWISE0:%.+]] = VPU.NCE.Eltwise([[QUERY0]], [[ARG2_0]])
      // CHECK-SAME: vf_loop_tile_index = 0 : i64
      // CHECK-SAME: -> tensor<1x1152x8x1xf16, {order = #NHWC}>
      // CHECK: [[SOFTMAX0:%.+]] = VPU.SoftMax([[ELTWISE0]])
      // CHECK-SAME: vf_loop_tile_index = 0 : i64
      // CHECK: [[PROB0:%.+]] = VPU.PermuteCast([[SOFTMAX0]])
      // CHECK: [[PROB_RESHAPE0:%.+]] = VPU.AffineReshape([[PROB0]])
      // CHECK: [[PROB_CAST0:%.+]] = VPU.PermuteCast([[PROB_RESHAPE0]])
      // CHECK: [[FILTER1_0:%.+]] = VPU.Slice %arg3 [0, 0, 0, 0, 0] [8, 96, 1152, 1, 1]
      // CHECK: [[MATMUL1_0:%.+]] = VPU.NCE.MatMul([[PROB_CAST0]], [[FILTER1_0]]) rawFilterShape [8, 96, 1152, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 0 : i64

      // CHECK: [[ARG0_1:%.+]] = VPU.Slice %arg0 [0, 8, 0, 0] [1, 8, 1, 96]
      // CHECK: [[FILTER0_1:%.+]] = VPU.Slice %arg1 [8, 0, 0, 0, 0] [8, 1152, 96, 1, 1]
      // CHECK: [[MATMUL0_1:%.+]] = VPU.NCE.MatMul({{%.+}}, [[FILTER0_1]]) rawFilterShape [8, 1152, 96, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 1 : i64
      // CHECK: [[ARG2_1:%.+]] = VPU.Slice %arg2 [0, 0, 8, 0] [1, 1152, 8, 1]
      // CHECK: [[ELTWISE1:%.+]] = VPU.NCE.Eltwise({{%.+}}, [[ARG2_1]])
      // CHECK-SAME: vf_loop_tile_index = 1 : i64
      // CHECK: [[SOFTMAX1:%.+]] = VPU.SoftMax([[ELTWISE1]])
      // CHECK-SAME: vf_loop_tile_index = 1 : i64
      // CHECK: [[FILTER1_1:%.+]] = VPU.Slice %arg3 [8, 0, 0, 0, 0] [8, 96, 1152, 1, 1]
      // CHECK: [[MATMUL1_1:%.+]] = VPU.NCE.MatMul({{%.+}}, [[FILTER1_1]]) rawFilterShape [8, 96, 1152, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 1 : i64

      // CHECK: [[ARG0_2:%.+]] = VPU.Slice %arg0 [0, 16, 0, 0] [1, 8, 1, 96]
      // CHECK: [[FILTER0_2:%.+]] = VPU.Slice %arg1 [16, 0, 0, 0, 0] [8, 1152, 96, 1, 1]
      // CHECK: [[MATMUL0_2:%.+]] = VPU.NCE.MatMul({{%.+}}, [[FILTER0_2]]) rawFilterShape [8, 1152, 96, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 2 : i64
      // CHECK: [[ARG2_2:%.+]] = VPU.Slice %arg2 [0, 0, 16, 0] [1, 1152, 8, 1]
      // CHECK: [[ELTWISE2:%.+]] = VPU.NCE.Eltwise({{%.+}}, [[ARG2_2]])
      // CHECK-SAME: vf_loop_tile_index = 2 : i64
      // CHECK: [[SOFTMAX2:%.+]] = VPU.SoftMax([[ELTWISE2]])
      // CHECK-SAME: vf_loop_tile_index = 2 : i64
      // CHECK: [[FILTER1_2:%.+]] = VPU.Slice %arg3 [16, 0, 0, 0, 0] [8, 96, 1152, 1, 1]
      // CHECK: [[MATMUL1_2:%.+]] = VPU.NCE.MatMul({{%.+}}, [[FILTER1_2]]) rawFilterShape [8, 96, 1152, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 2 : i64

      // CHECK: [[ARG0_3:%.+]] = VPU.Slice %arg0 [0, 24, 0, 0] [1, 8, 1, 96]
      // CHECK: [[FILTER0_3:%.+]] = VPU.Slice %arg1 [24, 0, 0, 0, 0] [8, 1152, 96, 1, 1]
      // CHECK: [[MATMUL0_3:%.+]] = VPU.NCE.MatMul({{%.+}}, [[FILTER0_3]]) rawFilterShape [8, 1152, 96, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 3 : i64
      // CHECK: [[ARG2_3:%.+]] = VPU.Slice %arg2 [0, 0, 24, 0] [1, 1152, 8, 1]
      // CHECK: [[ELTWISE3:%.+]] = VPU.NCE.Eltwise({{%.+}}, [[ARG2_3]])
      // CHECK-SAME: vf_loop_tile_index = 3 : i64
      // CHECK: [[SOFTMAX3:%.+]] = VPU.SoftMax([[ELTWISE3]])
      // CHECK-SAME: vf_loop_tile_index = 3 : i64
      // CHECK: [[FILTER1_3:%.+]] = VPU.Slice %arg3 [24, 0, 0, 0, 0] [8, 96, 1152, 1, 1]
      // CHECK: [[MATMUL1_3:%.+]] = VPU.NCE.MatMul({{%.+}}, [[FILTER1_3]]) rawFilterShape [8, 96, 1152, 1, 1]
      // CHECK-SAME: vf_loop_tile_index = 3 : i64

      // CHECK: [[CONCAT:%.+]] = VPU.Concat([[MATMUL1_0]], [[MATMUL1_1]], [[MATMUL1_2]], [[MATMUL1_3]])
      // CHECK: static_offsets =
      // CHECK-SAME: 0, 0, 0, 0, 0], [8, 0, 0, 0, 0], [16, 0, 0, 0, 0], [24, 0, 0, 0, 0]]
      // CHECK: -> tensor<32x1x96x1x1xf16, {order = #GNHWC}>
      // CHECK: return [[CONCAT]] : tensor<32x1x96x1x1xf16, {order = #GNHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [2, 3],
          numSlicedInputs = 2 : i64,
          numTensorInputs = 2 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = [array<i64: 2, -1>, array<i64: 3, 3>]>
  }
}

// CHECK-LABEL: @VfTilingGenericSwLayersOnC
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x960x1x640xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x960x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x960x1x1xf16, {order = #NHWC}>)
func.func @VfTilingGenericSwLayersOnC(%arg0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1: tensor<1x960x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x960x1x1xf16, {order = #NHWC}>) -> tensor<1x960x1x640xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1 as %a1: tensor<1x960x1x1xf16, {order = #NHWC}>, %arg2 as %a2: tensor<1x960x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x960x1x640xf16, {order = #NHWC}> {
    %inner0 = VPU.GenericSwLayer(%a0, %a1 : tensor<1x960x1x640xf16, {order = #NHWC}>, tensor<1x960x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [640, 960], offsets = [0, 0]) -> tensor<1x960x1x640xf16, {order = #NHWC}>
    %inner1 = VPU.GenericSwLayer(%inner0, %a2 : tensor<1x960x1x640xf16, {order = #NHWC}>, tensor<1x960x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [640, 960], offsets = [0, 0]) -> tensor<1x960x1x640xf16, {order = #NHWC}>
    VPU.Yield %inner1
  }
  return %0 : tensor<1x960x1x640xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 480, 1, 640]
    // CHECK-SAME:      : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 480, 1, 1]
    // CHECK-SAME:      : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x480x1x1xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[SLICE1]]
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 480], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT2]] [0, 0, 0, 0] [1, 480, 1, 1]
    // CHECK-SAME:      : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x480x1x1xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[GSL0]], [[SLICE2]]
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 480], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT0]] [0, 480, 0, 0] [1, 480, 1, 640]
    // CHECK-SAME:      : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[INPUT1]] [0, 480, 0, 0] [1, 480, 1, 1]
    // CHECK-SAME:      : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x480x1x1xf16, {order = #NHWC}>
    // CHECK:       [[GSL2:%.+]] = VPU.GenericSwLayer([[SLICE3]], [[SLICE4]]
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 480], offsets = [0, 480])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[INPUT2]] [0, 480, 0, 0] [1, 480, 1, 1]
    // CHECK-SAME:      : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x480x1x1xf16, {order = #NHWC}>
    // CHECK:       [[GSL3:%.+]] = VPU.GenericSwLayer([[GSL2]], [[SLICE5]]
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 480], offsets = [0, 480])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GSL1]], [[GSL3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 480, 0, 0]]}
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x960x1x640xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x960x1x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Same kernel, but tiled on mem dim 2 (logical W).

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [2, 3],
          numSlicedInputs = 2 : i64,
          numTensorInputs = 2 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = [array<i64: 2, -1>, array<i64: 3, 3>]>
  }
}

// CHECK-LABEL: @VfTilingGenericSwLayersOnW
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x640x1x960xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>)
func.func @VfTilingGenericSwLayersOnW(%arg0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1: tensor<1x640x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x640x1x1xf16, {order = #NHWC}>) -> tensor<1x640x1x960xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1 as %a1: tensor<1x640x1x1xf16, {order = #NHWC}>, %arg2 as %a2: tensor<1x640x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x640x1x960xf16, {order = #NHWC}> {
    %inner0 = VPU.GenericSwLayer(%a0, %a1 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) -> tensor<1x640x1x960xf16, {order = #NHWC}>
    %inner1 = VPU.GenericSwLayer(%inner0, %a2 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) -> tensor<1x640x1x960xf16, {order = #NHWC}>
    VPU.Yield %inner1
  }
  return %0 : tensor<1x640x1x960xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 640, 1, 480]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[GSL0]], [[INPUT2]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 480] [1, 640, 1, 480]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL2:%.+]] = VPU.GenericSwLayer([[SLICE1]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [480, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL3:%.+]] = VPU.GenericSwLayer([[GSL2]], [[INPUT2]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [480, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GSL1]], [[GSL3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 480]]}
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x640x1x960xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x640x1x960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<()[s0] -> (960, s0)>
#map1 = affine_map<()[s0, s1] -> (-s0 + 960, s1)>

// The kernel pads mem dim 3 (logical C) from 960 up to 1000, so the slice-info function clamps the
// input slice with affine.min. The tiles are therefore unequal: the second one consumes 448 input
// channels while producing 488 output channels.

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = affine.min #map()[%arg3]
    %1 = affine.min #map1()[%0, %arg1]
    return %c1, %c1, %arg0, %1, %c0, %c0, %arg2, %0, %c1, %c1, %c1, %1, %c0, %c0, %c0, %0 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [2, 3],
          numSlicedInputs = 2 : i64,
          numTensorInputs = 2 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = [array<i64: 2, -1>, array<i64: 3, 3>]>
  }
}

// CHECK-LABEL: @VfTilingPaddedGenericSwLayer
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x960x1x640xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x960x1x1xf16, {order = #NHWC}>)
func.func @VfTilingPaddedGenericSwLayer(%arg0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1: tensor<1x960x1x1xf16, {order = #NHWC}>) -> tensor<1x1000x1x640xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x960x1x640xf16, {order = #NHWC}>, %arg1 as %a1: tensor<1x960x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x1000x1x640xf16, {order = #NHWC}> {
    %inner = VPU.GenericSwLayer(%a0, %a1 : tensor<1x960x1x640xf16, {order = #NHWC}>, tensor<1x960x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [640, 1000], offsets = [0, 0]) -> tensor<1x1000x1x640xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  return %0 : tensor<1x1000x1x640xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 512, 1, 640]
    // CHECK-SAME:      : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x512x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 512, 1, 1]
    // CHECK-SAME:      : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[SLICE1]]
    // CHECK-SAME:      : tensor<1x512x1x640xf16, {order = #NHWC}>, tensor<1x512x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 512], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x512x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 512, 0, 0] [1, 448, 1, 640]
    // CHECK-SAME:      : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x448x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT1]] [0, 512, 0, 0] [1, 448, 1, 1]
    // CHECK-SAME:      : tensor<1x960x1x1xf16, {order = #NHWC}> to tensor<1x448x1x1xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE2]], [[SLICE3]]
    // CHECK-SAME:      : tensor<1x448x1x640xf16, {order = #NHWC}>, tensor<1x448x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 488], offsets = [0, 512])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x488x1x640xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 512, 0, 0]]}
    // CHECK-SAME:      : tensor<1x512x1x640xf16, {order = #NHWC}>, tensor<1x488x1x640xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x1000x1x640xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x1000x1x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %c1, %arg0, %arg1,
            %c0, %c0, %arg2, %arg3,
           %c1, %c1, %c1, %arg1,
            %c0, %c0, %c0, %arg3,
           %c1, %c1, %arg0, %arg1,
            %c0, %c0, %arg2, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: tensor<1x1x?x?xf16>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [2, 3],
          numSlicedInputs = 2 : i64,
          numTensorInputs = 2 : i64,
          numAuxBuffers = 1 : i64,
          inputSegmentationAxes = [array<i64: 2, -1, 2>, array<i64: 3, 3, 3>]>
  }
}

// CHECK-LABEL: @VfTilingGenericSwLayerWithScratch
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x640x1x960xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>)
func.func @VfTilingGenericSwLayerWithScratch(%arg0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1: tensor<1x640x1x1xf16, {order = #NHWC}>) -> tensor<1x640x1x960xf16, {order = #NHWC}> {
  %scratch = VPU.Empty : tensor<1x640x1x960xf16, {order = #NHWC}>
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1 as %a1: tensor<1x640x1x1xf16, {order = #NHWC}>, %scratch as %a2: tensor<1x640x1x960xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 4]} -> tensor<1x640x1x960xf16, {order = #NHWC}> {
    %inner = VPU.GenericSwLayer(%a0, %a1 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>)
               scratch(%a2 : tensor<1x640x1x960xf16, {order = #NHWC}>)
               @VPU.SW::@generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) -> tensor<1x640x1x960xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  return %0 : tensor<1x640x1x960xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SCRATCH:%.+]] = VPU.Empty : tensor<1x640x1x960xf16, {order = #NHWC}>
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SCRATCH0:%.+]] = VPU.Slice [[SCRATCH]] [0, 0, 0, 0] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) scratch([[SCRATCH0]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [240, 640], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 240] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SCRATCH1:%.+]] = VPU.Slice [[SCRATCH]] [0, 0, 0, 240] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE1]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) scratch([[SCRATCH1]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [240, 640], offsets = [240, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 480] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SCRATCH2:%.+]] = VPU.Slice [[SCRATCH]] [0, 0, 0, 480] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[GSL2:%.+]] = VPU.GenericSwLayer([[SLICE2]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) scratch([[SCRATCH2]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [240, 640], offsets = [480, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 2 : i64
    // CHECK-SAME:      -> tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 720] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[SCRATCH3:%.+]] = VPU.Slice [[SCRATCH]] [0, 0, 0, 720] [1, 640, 1, 240]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[GSL3:%.+]] = VPU.GenericSwLayer([[SLICE3]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) scratch([[SCRATCH3]]
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [240, 640], offsets = [720, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 3 : i64
    // CHECK-SAME:      -> tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]], [[GSL2]], [[GSL3]])
    // CHECK-SAME:      {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 240], [0, 0, 0, 480], [0, 0, 0, 720]]}
    // CHECK-SAME:      : tensor<1x640x1x240xf16, {order = #NHWC}>, tensor<1x640x1x240xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x640x1x240xf16, {order = #NHWC}>, tensor<1x640x1x240xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x640x1x960xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x640x1x960xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3 : index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [2, 3],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = [array<i64: 2>, array<i64: 3>]>
  }
}

// CHECK-LABEL: @VfTilingGenericSwLayerWithSwOp
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x960x1x640xf16, {order = #NHWC}>)
func.func @VfTilingGenericSwLayerWithSwOp(%arg0: tensor<1x960x1x640xf16, {order = #NHWC}>) -> tensor<1x960x1x640xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x960x1x640xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x960x1x640xf16, {order = #NHWC}> {
    %inner0 = VPU.GenericSwLayer(%a0 : tensor<1x960x1x640xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [640, 960], offsets = [0, 0]) -> tensor<1x960x1x640xf16, {order = #NHWC}>
    %inner1 = VPU.Sigmoid(%inner0) : tensor<1x960x1x640xf16, {order = #NHWC}> -> tensor<1x960x1x640xf16, {order = #NHWC}>
    VPU.Yield %inner1
  }
  return %0 : tensor<1x960x1x640xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 480, 1, 640]
    // CHECK-SAME:      : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]] : tensor<1x480x1x640xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 480], offsets = [0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SIGMOID0:%.+]] = VPU.Sigmoid([[GSL0]])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}> -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT0]] [0, 480, 0, 0] [1, 480, 1, 640]
    // CHECK-SAME:      : tensor<1x960x1x640xf16, {order = #NHWC}> to tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[SLICE1]] : tensor<1x480x1x640xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [640, 480], offsets = [0, 480])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[SIGMOID1:%.+]] = VPU.Sigmoid([[GSL1]])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}> -> tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[SIGMOID0]], [[SIGMOID1]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 480, 0, 0]]}
    // CHECK-SAME:      : tensor<1x480x1x640xf16, {order = #NHWC}>, tensor<1x480x1x640xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x960x1x640xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x960x1x640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> (index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %arg0, %arg1, %arg2, %c0, %arg3, %arg4, %arg5 : index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x?x?x?xf16>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) -> tensor<1x?x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [1, 2, 3],
          numSlicedInputs = 1 : i64,
          numTensorInputs = 1 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = [array<i64: 1>, array<i64: 2>, array<i64: 3>]>
  }
}

// CHECK-LABEL: @VfTilingConvPlusGenericSwLayer
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x16x128x128xf16, {order = #NHWC}>, [[WEIGHTS:%.+]]: tensor<32x16x3x3xf16, {order = #NHWC}>)
func.func @VfTilingConvPlusGenericSwLayer(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>, %wt: tensor<32x16x3x3xf16, {order = #NHWC}>) -> tensor<1x32x128x128xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x16x128x128xf16, {order = #NHWC}>, %wt as %a1: tensor<32x16x3x3xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x32x128x128xf16, {order = #NHWC}> {
    %conv = VPU.NCE.Convolution(%a0, %a1) rawFilterShape [32, 16, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>, strides = [1, 1]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %gsl = VPU.GenericSwLayer(%conv : tensor<1x32x128x128xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [128, 128, 32], offsets = [0, 0, 0]) -> tensor<1x32x128x128xf16, {order = #NHWC}>
    VPU.Yield %gsl
  }
  return %0 : tensor<1x32x128x128xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 16, 33, 128]
    // CHECK-SAME:      : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x33x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[WEIGHTS]]) rawFilterShape [32, 16, 3, 3] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 0 : i64>
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[CONV0]] : tensor<1x32x32x128xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [32, 128, 32], offsets = [0, 0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 31, 0] [1, 16, 34, 128]
    // CHECK-SAME:      : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x34x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]]) rawFilterShape [32, 16, 3, 3] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[CONV1]] : tensor<1x32x32x128xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [32, 128, 32], offsets = [32, 0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 63, 0] [1, 16, 34, 128]
    // CHECK-SAME:      : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x34x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]]) rawFilterShape [32, 16, 3, 3] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      vf_loop_tile_index = 2 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>
    // CHECK:       [[GSL2:%.+]] = VPU.GenericSwLayer([[CONV2]] : tensor<1x32x32x128xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [32, 128, 32], offsets = [64, 0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 2 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>

    // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 95, 0] [1, 16, 33, 128]
    // CHECK-SAME:      : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x33x128xf16, {order = #NHWC}>
    // CHECK:       [[CONV3:%.+]] = VPU.NCE.Convolution([[SLICE3]], [[WEIGHTS]]) rawFilterShape [32, 16, 3, 3] {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>
    // CHECK-SAME:      vf_loop_tile_index = 3 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>
    // CHECK:       [[GSL3:%.+]] = VPU.GenericSwLayer([[CONV3]] : tensor<1x32x32x128xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [32, 128, 32], offsets = [96, 0, 0])
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 3 : i64
    // CHECK-SAME:      -> tensor<1x32x32x128xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GSL0]], [[GSL1]], [[GSL2]], [[GSL3]])
    // CHECK-SAME:      {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 32, 0], [0, 0, 64, 0], [0, 0, 96, 0]]}
    // CHECK-SAME:      -> tensor<1x32x128x128xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x32x128x128xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VPU.SW {
  func.func @generated_info_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    return %c1, %c1, %arg0, %arg1, %c0, %c0, %arg2, %arg3, %c1, %c1, %c1, %arg1, %c0, %c0, %c0, %arg3 : index, index, index, index, index, index, index, index, index, index, index, index, index, index, index, index
  }
  func.func nested @generated_0(%arg0: tensor<1x1x?x?xf16>, %arg1: tensor<1x1x1x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<1x1x?x?xf16> attributes {
      kernelInfo = #VPU.KernelInfo<
          tilingInfoFunc = @generated_info_0,
          tilingAxes = [2, 3],
          numSlicedInputs = 2 : i64,
          numTensorInputs = 2 : i64,
          numAuxBuffers = 0 : i64,
          inputSegmentationAxes = [array<i64: 2, -1>, array<i64: 3, 3>]>
  }
}

// CHECK-LABEL: @VfTilingGenericSwLayersWithMCStrategy
// CHECK-SAME:      [[INPUT0:%.+]]: tensor<1x640x1x960xf16, {order = #NHWC}>, [[INPUT1:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%.+]]: tensor<1x640x1x1xf16, {order = #NHWC}>)
func.func @VfTilingGenericSwLayersWithMCStrategy(%arg0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1: tensor<1x640x1x1xf16, {order = #NHWC}>, %arg2: tensor<1x640x1x1xf16, {order = #NHWC}>) -> tensor<1x640x1x960xf16, {order = #NHWC}> {
  %0 = VPU.VerticalFusion (%arg0 as %a0: tensor<1x640x1x960xf16, {order = #NHWC}>, %arg1 as %a1: tensor<1x640x1x1xf16, {order = #NHWC}>, %arg2 as %a2: tensor<1x640x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x640x1x960xf16, {order = #NHWC}> {
    %inner0 = VPU.GenericSwLayer(%a0, %a1 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} -> tensor<1x640x1x960xf16, {order = #NHWC}>
    %inner1 = VPU.GenericSwLayer(%inner0, %a2 : tensor<1x640x1x960xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0 tiling(sizes = [960, 640], offsets = [0, 0]) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} -> tensor<1x640x1x960xf16, {order = #NHWC}>
    VPU.Yield %inner1
  }
  return %0 : tensor<1x640x1x960xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.VerticalFusion
    // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 0] [1, 640, 1, 480]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL0:%.+]] = VPU.GenericSwLayer([[SLICE0]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [0, 0])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL1:%.+]] = VPU.GenericSwLayer([[GSL0]], [[INPUT2]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [0, 0])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 0 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT0]] [0, 0, 0, 480] [1, 640, 1, 480]
    // CHECK-SAME:      : tensor<1x640x1x960xf16, {order = #NHWC}> to tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL2:%.+]] = VPU.GenericSwLayer([[SLICE1]], [[INPUT1]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [480, 0])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[GSL3:%.+]] = VPU.GenericSwLayer([[GSL2]], [[INPUT2]]
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x1xf16, {order = #NHWC}>) @VPU.SW::@generated_0
    // CHECK-SAME:      tiling(sizes = [480, 640], offsets = [480, 0])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:      vf_loop_index = 0 : i64
    // CHECK-SAME:      vf_loop_tile_index = 1 : i64
    // CHECK-SAME:      -> tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[GSL1]], [[GSL3]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 480]]}
    // CHECK-SAME:      : tensor<1x640x1x480xf16, {order = #NHWC}>, tensor<1x640x1x480xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x640x1x960xf16, {order = #NHWC}>
    // CHECK:       return [[CONCAT]] : tensor<1x640x1x960xf16, {order = #NHWC}>
}

