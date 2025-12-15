//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --merge-vertical-fusion-subgraphs="enable-vertical-fusion-pipelining=true workload-management-mode=PWLM_V1_BARRIER_FIFO" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildSubgraphWith2dTiling(%arg0: tensor<1x128x512x512xf16, {order = #NHWC}>) -> tensor<1x128x512x512xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<128x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>
    %cst_2 = const.Declare tensor<128x128x3x3xf16, {order = #NHWC}> = dense<2.0> : tensor<128x128x3x3xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<128x1x1x4xsi32> = dense<2> : tensor<128x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x512x512xf16, {order = #NHWC}>,
                             %cst_0 as %arg2: tensor<128x16x1x1xf16, {order = #NHWC}>,
                             %cst_1 as %arg3: tensor<128x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 64]} -> tensor<1x128x512x512xf16, {order = #NHWC}> {
        %inner = VPU.NCE.DepthConvolution(%arg1, %arg2, %arg3)
                       {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                        ppe = #VPU.PPEFp<mode = <SWISH>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                        rawFilterShape = [128, 1, 1, 1], strides = [1, 1]} -> tensor<1x128x512x512xf16, {order = #NHWC}>
        VPU.Yield %inner
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x128x512x512xf16, {order = #NHWC}>,
                             %cst_2 as %arg2: tensor<128x128x3x3xf16, {order = #NHWC}>,
                             %cst_3 as %arg3: tensor<128x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 43]} -> tensor<1x128x512x512xf16, {order = #NHWC}> {
        %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
                       {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                       multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                       pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                       ppe = #VPU.PPEFp<mode = <NOOP>,
                       clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                       prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
                       rawFilterShape = [128, 128, 3, 3], strides = [1, 1]} : tensor<1x128x512x512xf16, {order = #NHWC}>, tensor<128x128x3x3xf16, {order = #NHWC}>, tensor<128x1x1x4xsi32> -> tensor<1x128x512x512xf16, {order = #NHWC}>
        VPU.Yield %inner
    }

    return %1 : tensor<1x128x512x512xf16, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 11, 9]
    //CHECK:      [[DWCONV:%.+]] = VPU.NCE.DepthConvolution
    //CHECK:      [[CONV:%.+]] = VPU.NCE.Convolution([[DWCONV]]
    //CHECK:      return [[VERTICAL_FUSION]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.013744638480392157:128>
!qElemType1 = !quant.uniform<u8:f16:0, {0.0038832720588235295:128,0.0031929764093137254:128,0.0036142386642156864:128,0.0036563648897058824:128,0.0035060508578431374:128,0.0039905024509803919:128,0.0036659390318627451:128,0.0031968060661764705:128,0.0035213694852941177:128,0.0032619102328431374:128,0.0038411458333333331:128,0.0035251991421568628:128,0.003833486519607843:128,0.003372012867647059:128,0.0035816865808823528:128,0.0037023207720588234:128,0.0038200827205882352:128,0.0036123238357843139:128,0.003345205269607843:128,0.0031163832720588237:128,0.0036506204044117647:128,0.0034888174019607845:128,0.0038736979166666668:128,0.0033758425245098041:128,0.003058938419117647:128,0.0037176393995098037:128,0.0034562653186274508:128,0.0033260569852941175:128,0.003349034926470588:128,0.0041475183823529412:128,0.0041207107843137256:128,0.003490732230392157:128}>

func.func @BuildSubgraphRollBackTo1dTiling(%arg0: tensor<1x16x256x256x!qElemType, {order = #NHWC}>) -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x16x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x16x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x32x3x3x!qElemType1, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType1>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x16x256x256x!qElemType, {order = #NHWC}>, %cst_0 as %arg2: tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, %cst_1 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 2]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 16, 3, 3], strides = [1, 1]} : tensor<1x16x256x256x!qElemType, {order = #NHWC}>, tensor<32x16x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, %cst_3 as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 2, 2]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
        {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [32, 32, 3, 3], strides = [1, 1]} : tensor<1x32x256x256x!qElemType, {order = #NHWC}>, tensor<32x32x3x3x!qElemType1, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }
    %2 = VPU.VerticalFusion (%0 as %arg1: tensor<1x32x256x256x!qElemType, {order = #NHWC}>, %1 as %arg2: tensor<1x32x256x256x!qElemType, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 2]} -> tensor<1x32x256x256x!qElemType, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2)
         {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>,
         ppe = #VPU.PPEStub<>}
         -> tensor<1x32x256x256x!qElemType, {order = #NHWC}>
      VPU.Yield %3
    }

    return %2 : tensor<1x32x256x256x!qElemType, {order = #NHWC}>


    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 1, 3]
    //CHECK:      [[CONV0:%.+]] = VPU.NCE.Convolution
    //CHECK:      [[CONV1:%.+]] = VPU.NCE.Convolution([[CONV0]]
    //CHECK:      [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV0]], [[CONV1]])
    //CHECK:        VPU.Yield [[ELTWISE]]

    //CHECK: return [[VERTICAL_FUSION]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @BuildSubgraphKeep2dTiling(%arg0: tensor<1x16x512x4096xf16, {order = #NHWC}>, %arg1: tensor<1x16x512x4096xf16, {order = #NHWC}>, %arg2: tensor<1x16x512x4096xf16, {order = #NHWC}>) -> tensor<1x16x512x4096xf16, {order = #NHWC}> {
   %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x16x512x4096xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<1x16x512x4096xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 74, 2]} -> tensor<1x16x512x4096xf16, {order = #NHWC}> {
     %2 = VPU.NCE.Eltwise(%arg3, %arg4) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x512x4096xf16, {order = #NHWC}>
     VPU.Yield %2
   }
   %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x16x512x4096xf16, {order = #NHWC}>, %arg2 as %arg4: tensor<1x16x512x4096xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 74, 2]} -> tensor<1x16x512x4096xf16, {order = #NHWC}> {
     %2 = VPU.NCE.Eltwise(%arg3, %arg4) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x16x512x4096xf16, {order = #NHWC}>
     VPU.Yield %2
   }
   return %1 : tensor<1x16x512x4096xf16, {order = #NHWC}>

    //CHECK:      [[VERTICAL_FUSION:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 1, 32, 10]
    //CHECK:      [[ELTWISE0:%.+]] = VPU.NCE.Eltwise
    //CHECK:      [[ELTWISE1:%.+]] = VPU.NCE.Eltwise([[ELTWISE0]]
    //CHECK:        VPU.Yield [[ELTWISE1]]

    //CHECK: return [[VERTICAL_FUSION]]
 }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
//CHECK-LABEL: @BuildSubgraphWithSharedWeights
//CHECK-SAME:  [[INPUT:%.+]]: tensor<1x160x2000x4xf16, {order = #NHWC}>
func.func @BuildSubgraphWithSharedWeights(%arg0: tensor<1x160x2000x4xf16, {order = #NHWC}>) -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
  %w_0= const.Declare tensor<256x160x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<256x160x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %wt_0 = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>
  %w_1 = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<256x256x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %wt_1 = const.Declare tensor<256x1x1x4xsi32> = dense<2> : tensor<256x1x1x4xsi32>
  %w_2 = const.Declare tensor<256x160x1x1xf16, {order = #NHWC}> = dense<3.0> : tensor<256x160x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %wt_2 = const.Declare tensor<256x1x1x4xsi32> = dense<3> : tensor<256x1x1x4xsi32>
  %w_3 = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<4.0> : tensor<256x256x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %wt_3 = const.Declare tensor<256x1x1x4xsi32> = dense<4> : tensor<256x1x1x4xsi32>
  %w_4 = const.Declare tensor<256x256x1x1xf16, {order = #NHWC}> = dense<5.0> : tensor<256x256x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
  %wt_4 = const.Declare tensor<256x1x1x4xsi32> = dense<5> : tensor<256x1x1x4xsi32>

  %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x160x2000x4xf16, {order = #NHWC}>,
                           %w_0 as %arg2: tensor<256x160x1x1xf16, {order = #NHWC}>,
                           %wt_0 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {input_padding = [0, 2, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [0.0999755859375], adder = 0.000000e+00 : f64>, rawFilterShape = [256, 160, 1, 1], strides = [1, 1]} : tensor<1x160x2000x4xf16, {order = #NHWC}>, tensor<256x160x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x2000x4xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %1 = VPU.VerticalFusion (%0 as %arg1: tensor<1x256x2000x4xf16, {order = #NHWC}>,
                           %w_1 as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
                           %wt_1 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} : tensor<1x256x2000x4xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x2000x4xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %2 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x160x2000x4xf16, {order = #NHWC}>,
                           %w_2 as %arg2: tensor<256x160x1x1xf16, {order = #NHWC}>,
                           %wt_2 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {input_padding = [0, 2, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [0.0999755859375], adder = 0.000000e+00 : f64>, rawFilterShape = [256, 160, 1, 1], strides = [1, 1]} : tensor<1x160x2000x4xf16, {order = #NHWC}>, tensor<256x160x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x2000x4xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x256x2000x4xf16, {order = #NHWC}>,
                           %w_3 as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
                           %wt_3 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} : tensor<1x256x2000x4xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x2000x4xf16, {order = #NHWC}>
    VPU.Yield %inner
  }

  %4 = VPU.VerticalFusion (%3 as %arg1: tensor<1x256x2000x4xf16, {order = #NHWC}>,
                           %1 as %arg2: tensor<1x256x2000x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [0.0999755859375], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x256x2000x4xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  %5 = VPU.VerticalFusion (%4 as %arg1: tensor<1x256x2000x4xf16, {order = #NHWC}>,
                           %w_4 as %arg2: tensor<256x256x1x1xf16, {order = #NHWC}>,
                           %wt_4 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 4, 1]} -> tensor<1x256x2000x4xf16, {order = #NHWC}> {
    %inner = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [256, 256, 1, 1], strides = [1, 1]} : tensor<1x256x2000x4xf16, {order = #NHWC}>, tensor<256x256x1x1xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x2000x4xf16, {order = #NHWC}>
    VPU.Yield %inner
  }
  return %5 : tensor<1x256x2000x4xf16, {order = #NHWC}>

  //CHECK:  [[VF:%.+]] = VPU.VerticalFusion ([[INPUT]]
  // Without considering the shared weights, the tiling strategy could be [1, 1, 6, 1]
  //CHECK-SAME: tilingStrategy = [1, 1, 7, 1]}
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  VPU.NCE.Convolution
  //CHECK:  VPU.NCE.Eltwise
  //CHECK:  return [[VF]] : tensor<1x256x2000x4xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @MergeVFWithPipelinedUser
// CHECK-SAME:      tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      tensor<8192x128x1x1xf16, {order = #NHWC}>,
// CHECK-SAME:      tensor<1x8192x256x4xf16, {order = #NHWC}>)
func.func @MergeVFWithPipelinedUser(
    %arg0: tensor<1x128x256x4xf16, {order = #NHWC}>,
    %arg1: tensor<8192x128x1x1xf16, {order = #NHWC}>,
    %arg2: tensor<1x8192x256x4xf16, {order = #NHWC}>) -> tensor<1x8192x256x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<8192x1x1x4xsi32> = dense<1> : tensor<8192x1x1x4xsi32>
    %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<128x1x1x4xsi32>

    %0 = VPU.VerticalFusion (%arg0 as %arg3: tensor<1x128x256x4xf16, {order = #NHWC}>, %arg1 as %arg4: tensor<8192x128x1x1xf16, {order = #NHWC}>, %cst_0 as %arg5: tensor<8192x1x1x4xsi32>) attributes {tilingStrategy = [1, 16, 1, 1]} -> tensor<1x8192x256x4xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg3, %arg4, %arg5) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [8192, 128, 1, 1], strides = [1, 1]} : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<8192x128x1x1xf16, {order = #NHWC}>, tensor<8192x1x1x4xsi32> -> tensor<1x8192x256x4xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    %1 = VPU.VerticalFusion (%0 as %arg3: tensor<1x8192x256x4xf16, {order = #NHWC}>, %arg2 as %arg4: tensor<1x8192x256x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 8, 1, 1]} -> tensor<1x8192x256x4xf16, {order = #NHWC}> {
      %4 = VPU.NCE.Eltwise(%arg3, %arg4) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x8192x256x4xf16, {order = #NHWC}> 
      VPU.Yield %4
    }
    %2 = VPU.VerticalFusion (%1 as %arg3: tensor<1x8192x256x4xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 8, 1]} -> tensor<1x8192x256x4xf16, {order = #NHWC}> {
      %5 = VPU.SoftMax(%arg3) {axisInd = 1 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x8192x256x4xf16, {order = #NHWC}> -> tensor<1x8192x256x4xf16, {order = #NHWC}>
      VPU.Yield %5
    }

    return %2 : tensor<1x8192x256x4xf16, {order = #NHWC}>


    //CHECK: [[VF:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: tilingStrategy = [1, 16, 1, 1]
    //CHECK:    [[CONV:%.+]] = VPU.NCE.Convolution

    //CHECK: [[VF1:%.+]] = VPU.VerticalFusion
    //CHECK-SAME: scenario = #VPU.vf_scenario<VF_PIPELINING>
    //CHECK-SAME: tilingStrategy = [1, 1, 17, 1]
    //CHECK:    [[ELTWISE:%.+]] = VPU.NCE.Eltwise
    //CHECK:    [[SOFTMAX:%.+]] = VPU.SoftMax

    //CHECK: return [[VF1]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MergeForLargeIteration
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x9x272x480xf32>
func.func @MergeForLargeIteration(%arg0: tensor<1x9x272x480xf32>) -> tensor<1x20x68x120xf32> {
    %cst = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %cst_0 = const.Declare tensor<64x1x1x4xsi32> = dense<2> : tensor<64x1x1x4xsi32>
    %cst_1 = const.Declare tensor<256x1x1x4xsi32> = dense<11> : tensor<256x1x1x4xsi32>
    %cst_2 = const.Declare tensor<32x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<20x64x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [12, 0, 0, 0]>]
    %cst_3 = const.Declare tensor<256x144x3x2xf16, {order = #NHWC}> = dense<1.0> : tensor<256x288x1x3xf16, {order = #NHWC}>, [#const.Reshape<[256, 144, 3, 2]>]
    %cst_4 = const.Declare tensor<1x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x1x1xf16>, [#const.Reshape<[1, 64, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<1x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x1x1xf16>, [#const.Reshape<[1, 32, 1, 1]>, #const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_6 = const.Declare tensor<64x64x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_7 = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<64x32x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x9x272x480xf32>) attributes {tilingStrategy = [1, 1, 1, 2]} -> tensor<1x9x272x480xf16> {
      %26 = VPU.Convert(%arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x9x272x480xf32> -> tensor<1x9x272x480xf16>
      VPU.Yield %26
    }
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 9 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 2]} -> tensor<1x9x272x480xf16, {order = #NHWC}> 
    %2 = VPU.ShapeCast {shape = [1, 144, 272, 30]} inputs(%1 : tensor<1x9x272x480xf16, {order = #NHWC}>) -> tensor<1x144x272x30xf16, {order = #NHWC}>
    %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x144x272x30xf16, {order = #NHWC}>, %cst_3 as %arg2: tensor<256x144x3x2xf16, {order = #NHWC}>, %cst_1 as %arg3: tensor<256x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 3]} -> tensor<1x256x136x30xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [256, 144, 3, 2], strides = [2, 1]} : tensor<1x144x272x30xf16, {order = #NHWC}>, tensor<256x144x3x2xf16, {order = #NHWC}>, tensor<256x1x1x4xsi32> -> tensor<1x256x136x30xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %4 = VPU.VerticalFusion (%3 as %arg1: tensor<1x256x136x30xf16, {order = #NHWC}>, %cst_5 as %arg2: tensor<1x32x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x136x240xf16, {order = #NHWC}> {
      %26 = VPU.ShapeCast {shape = [1, 32, 136, 240]} inputs(%arg1 : tensor<1x256x136x30xf16, {order = #NHWC}>) -> tensor<1x32x136x240xf16, {order = #NHWC}>
      %27 = VPU.PRelu(%26, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x32x136x240xf16, {order = #NHWC}>, tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x136x240xf16, {order = #NHWC}>
      VPU.Yield %27
    }
    %5 = VPU.VerticalFusion (%4 as %arg1: tensor<1x32x136x240xf16, {order = #NHWC}>, %cst_7 as %arg2: tensor<64x32x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 32, 3, 3], strides = [2, 2]} : tensor<1x32x136x240xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %6 = VPU.VerticalFusion (%5 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %7 = VPU.VerticalFusion (%6 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %8 = VPU.VerticalFusion (%7 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %9 = VPU.VerticalFusion (%8 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %10 = VPU.VerticalFusion (%9 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %11 = VPU.VerticalFusion (%10 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %12 = VPU.VerticalFusion (%11 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %13 = VPU.VerticalFusion (%12 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %14 = VPU.VerticalFusion (%13 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %15 = VPU.VerticalFusion (%14 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %16 = VPU.VerticalFusion (%15 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %17 = VPU.VerticalFusion (%16 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %18 = VPU.VerticalFusion (%17 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %19 = VPU.VerticalFusion (%18 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %20 = VPU.VerticalFusion (%19 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %21 = VPU.VerticalFusion (%20 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_6 as %arg2: tensor<64x64x3x3xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [64, 64, 3, 3], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<64x64x3x3xf16, {order = #NHWC}>, tensor<64x1x1x4xsi32> -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %22 = VPU.VerticalFusion (%21 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_4 as %arg2: tensor<1x64x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.PRelu(%arg1, %arg2) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x68x120xf16, {order = #NHWC}>
      VPU.Yield %26
    }
    %23 = VPU.VerticalFusion (%22 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %6 as %arg2: tensor<1x64x68x120xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x68x120xf16, {order = #NHWC}> {
      %26 = VPU.NCE.Eltwise(%arg1, %arg2) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x64x68x120xf16, {order = #NHWC}> 
      VPU.Yield %26
    }
    %24 = VPU.VerticalFusion (%23 as %arg1: tensor<1x64x68x120xf16, {order = #NHWC}>, %cst_2 as %arg2: tensor<32x64x1x1xf16, {order = #NHWC}>, %cst as %arg3: tensor<32x1x1x4xsi32>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x32x68x120xf32> {
      %26 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {input_padding = [0, 0, 0, 0], mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, output_padding = [0, 12, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [32, 64, 1, 1], strides = [1, 1]} : tensor<1x64x68x120xf16, {order = #NHWC}>, tensor<32x64x1x1xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x68x120xf32> 
      VPU.Yield %26
    }
    %25 = VPU.Slice %24 [0, 0, 0, 0] [1, 20, 68, 120] : tensor<1x32x68x120xf32> to tensor<1x20x68x120xf32>
    return %25 : tensor<1x20x68x120xf32>


    // CHECK:      [[CONVERT:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]
    // CHECK:        VPU.Convert

    // CHECK:      [[CONV_PRELU_1:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.ShapeCast
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_2:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_3:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_4:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_5:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_6:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_7:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_8:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_9:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu

    // CHECK:      [[CONV_PRELU_10:%.+]] = VPU.VerticalFusion
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK:        VPU.NCE.Convolution
    // CHECK:        VPU.PRelu
    // CHECK:        VPU.NCE.Eltwise
    // CHECK:        VPU.NCE.Convolution

    // CHECK:      [[SLICE:%.+]] = VPU.Slice [[CONV_PRELU_10]]
    // CHECK:      return [[SLICE]]
}
