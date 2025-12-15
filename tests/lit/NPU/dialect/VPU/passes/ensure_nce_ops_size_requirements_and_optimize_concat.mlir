//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --ensure-nce-ops-size-requirements --optimize-concat --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvWithEltwiseAddOverOC
// CHECK-SAME:    [[INPUT0:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:    [[INPUT1:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
func.func @SplitNCEConvWithEltwiseAddOverOC(%input1: tensor<1x128x256x4xf16, {order = #NHWC}>, %input2: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x18944x256x4xf16, {order = #NHWC}> {
    %weights0 = const.Declare tensor<18944x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<18944x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weightsTable0 = const.Declare tensor<18944x1x1x4xsi32> = dense<2> : tensor<18944x1x1x4xsi32>
    %conv0 = VPU.NCE.Convolution(%input1, %weights0, %weightsTable0) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [18944, 128, 1, 1], strides = [1, 1]
    } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<18944x128x1x1x!qElemType, {order = #NHWC}>, tensor<18944x1x1x4xsi32> -> tensor<1x18944x256x4xf16, {order = #NHWC}>

    %weights1 = const.Declare tensor<18944x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<18944x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %weightsTable1 = const.Declare tensor<18944x1x1x4xsi32> = dense<2> : tensor<18944x1x1x4xsi32>
    %conv1 = VPU.NCE.Convolution(%input2, %weights1, %weightsTable1) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        rawFilterShape = [18944, 128, 1, 1], strides = [1, 1]
    } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<18944x128x1x1x!qElemType, {order = #NHWC}>, tensor<18944x1x1x4xsi32> -> tensor<1x18944x256x4xf16, {order = #NHWC}>

    %add = VPU.NCE.Eltwise(%conv0, %conv1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
        lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
        quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x18944x256x4xf16, {order = #NHWC}>

    return %add : tensor<1x18944x256x4xf16, {order = #NHWC}>

  //CHECK-DAG:    [[CST:%.+]] = const.Declare tensor<6304x1x1x4xsi32> = dense<2> : tensor<18944x1x1x4xsi32>, [#const.SubView<[12640, 0, 0, 0], [6304, 1, 1, 4]>]
  //CHECK-DAG:    [[CST_0:%.+]] = const.Declare tensor<6304x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<18944x128x1x1xf16>, [#const.SubView<[12640, 0, 0, 0], [6304, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK-DAG:    [[CST_1:%.+]] = const.Declare tensor<6320x1x1x4xsi32> = dense<2> : tensor<18944x1x1x4xsi32>, [#const.SubView<[6320, 0, 0, 0], [6320, 1, 1, 4]>]
  //CHECK-DAG:    [[CST_2:%.+]] = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<18944x128x1x1xf16>, [#const.SubView<[6320, 0, 0, 0], [6320, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK-DAG:    [[CST_3:%.+]] = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<18944x128x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [6320, 128, 1, 1]>, #const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK-DAG:    [[CST_4:%.+]] = const.Declare tensor<6320x1x1x4xsi32> = dense<2> : tensor<18944x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [6320, 1, 1, 4]>]

  //CHECK:        [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST_3]], [[CST_4]]) {
  //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  //CHECK-SAME:       rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
  //CHECK-SAME:   }
  //CHECK-SAME:   -> tensor<1x6320x256x4xf16, {order = #NHWC}>

  //CHECK:        [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST_2]], [[CST_1]]) {
  //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  //CHECK-SAME:       rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
  //CHECK-SAME:   }
  //CHECK-SAME:   -> tensor<1x6320x256x4xf16, {order = #NHWC}>

  //CHECK:        [[CONV_2:%.+]] = VPU.NCE.Convolution([[INPUT0]], [[CST_0]], [[CST]]) {
  //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  //CHECK-SAME:       rawFilterShape = [6304, 128, 1, 1], strides = [1, 1]
  //CHECK-SAME:   }
  //CHECK-SAME:   -> tensor<1x6304x256x4xf16, {order = #NHWC}>

  //CHECK:        [[CONV_3:%.+]] = VPU.NCE.Convolution([[INPUT1]], [[CST_3]], [[CST_4]]) {
  //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  //CHECK-SAME:       rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
  //CHECK-SAME:   }
  //CHECK-SAME:   -> tensor<1x6320x256x4xf16, {order = #NHWC}>

  //CHECK:        [[CONV_4:%.+]] = VPU.NCE.Convolution([[INPUT1]], [[CST_2]], [[CST_1]]) {
  //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  //CHECK-SAME:       rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
  //CHECK-SAME:   }
  //CHECK-SAME:   -> tensor<1x6320x256x4xf16, {order = #NHWC}>

  //CHECK:        [[CONV_5:%.+]] = VPU.NCE.Convolution([[INPUT1]], [[CST_0]], [[CST]]) {
  //CHECK-SAME:       pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
  //CHECK-SAME:       rawFilterShape = [6304, 128, 1, 1], strides = [1, 1]
  //CHECK-SAME:   }
  //CHECK-SAME:   -> tensor<1x6304x256x4xf16, {order = #NHWC}>

  //CHECK:        [[ADD_0:%.+]] = VPU.NCE.Eltwise([[CONV_0]], [[CONV_3]]) {
  //CHECK-SAME:       op_type = #VPU.eltwise_type<ADD>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
  //CHECK-SAME:   } -> tensor<1x6320x256x4xf16, {order = #NHWC}>

  //CHECK:        [[ADD_1:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_4]]) {
  //CHECK-SAME:       op_type = #VPU.eltwise_type<ADD>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
  //CHECK-SAME:   } -> tensor<1x6320x256x4xf16, {order = #NHWC}>

  //CHECK:        [[ADD_2:%.+]] = VPU.NCE.Eltwise([[CONV_2]], [[CONV_5]]) {
  //CHECK-SAME:       op_type = #VPU.eltwise_type<ADD>,
  //CHECK-SAME:       ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
  //CHECK-SAME:       lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
  //CHECK-SAME:   } -> tensor<1x6304x256x4xf16, {order = #NHWC}>

  //CHECK:        [[CONCAT:%.+]] = VPU.Concat([[ADD_0]], [[ADD_1]], [[ADD_2]]) {
  //CHECK-SAME{LITERAL}:  static_offsets = [[0, 0, 0, 0], [0, 6320, 0, 0], [0, 12640, 0, 0]]
  //CHECK-SAME:   } : tensor<1x6320x256x4xf16, {order = #NHWC}>,
  //CHECK-SAME:       tensor<1x6320x256x4xf16, {order = #NHWC}>,
  //CHECK-SAME:       tensor<1x6304x256x4xf16, {order = #NHWC}> -> tensor<1x18944x256x4xf16, {order = #NHWC}>

  //CHECK:        return [[CONCAT]] : tensor<1x18944x256x4xf16, {order = #NHWC}>
}
