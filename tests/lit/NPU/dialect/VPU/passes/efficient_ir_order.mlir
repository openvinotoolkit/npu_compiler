//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --efficient-ir-order="enable-reorder-concat-branches=false" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EfficientEltwiseOrder
// CHECK-SAME:      ([[INPUT:%arg[0-9]]]: tensor<1x96x24x40xf16, {order = #NHWC}>)
func.func @EfficientEltwiseOrder(%arg0: tensor<1x96x24x40xf16, {order = #NHWC}>) -> tensor<1x96x24x40xf16, {order = #NHWC}> {
    %cst_1 = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<96x96x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_1) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} : tensor<1x96x24x40xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x24x40xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_1) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} : tensor<1x96x24x40xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x24x40xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_1) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [1, 1]} : tensor<1x96x24x40xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x24x40xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%0, %cst_1) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 1 : i64, top = 0 : i64, bottom = 1 : i64>, rawFilterShape = [96, 96, 3, 3], strides = [2, 2]} : tensor<1x96x24x40xf16, {order = #NHWC}>, tensor<96x96x3x3xf16, {order = #NHWC}> -> tensor<1x96x12x20xf16, {order = #NHWC}>
    %4 = VPU.Concat(%3, %3) {per_axis = #IE.Concat<axis = 2 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x96x12x20xf16, {order = #NHWC}>, tensor<1x96x12x20xf16, {order = #NHWC}> -> tensor<1x96x24x20xf16, {order = #NHWC}>
    %5 = VPU.Concat(%4, %4) {per_axis = #IE.Concat<axis = 3 : i64, offset = 1 : i64, stride = 2 : i64>} : tensor<1x96x24x20xf16, {order = #NHWC}>, tensor<1x96x24x20xf16, {order = #NHWC}> -> tensor<1x96x24x40xf16, {order = #NHWC}>

    %6 = VPU.NCE.Eltwise(%2, %5) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x96x24x40xf16, {order = #NHWC}>
    return %6: tensor<1x96x24x40xf16, {order = #NHWC}>


    //CHECK:  [[WEIGHTS:%.+]] = const.Declare tensor<96x96x3x3xf16, {order = #NHWC}>
    //CHECK:  [[CONV_0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
    //CHECK:  [[CONV_1:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[WEIGHTS]])
    //CHECK:  [[CONCAT_0:%.+]] = VPU.Concat([[CONV_1]], [[CONV_1]])
    //CHECK:  [[CONCAT_1:%.+]] = VPU.Concat([[CONCAT_0]], [[CONCAT_0]])
    //CHECK:  [[CONV_2:%.+]] = VPU.NCE.Convolution([[CONV_0]], [[WEIGHTS]])
    //CHECK:  [[CONV_3:%.+]] = VPU.NCE.Convolution([[CONV_2]], [[WEIGHTS]])
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[CONV_3]], [[CONCAT_1]])
    //CHECK:  return [[ELTWISE]] : tensor<1x96x24x40xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EfficientGPTQOperationsOrder
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT2:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
func.func @EfficientGPTQOperationsOrder(
            %arg0: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %arg1: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %arg2: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x3584x256x4xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<3584x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<3584x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<3584x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<3584x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<3584x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> :
        tensor<3584x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (
            %arg0 as %arg3: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %cst_0 as %arg4: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
            %arg1 as %arg6: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %cst_2 as %arg7: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
            %arg2 as %arg9: tensor<1x128x256x4xf16, {order = #NHWC}>,
            %cst_4 as %arg10: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>
            ) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x3584x256x4xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%arg3, %arg4) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [3584, 128, 1, 1], strides = [1, 1]
        } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
        %2 = VPU.NCE.Convolution(%arg6, %arg7) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [3584, 128, 1, 1], strides = [1, 1]
        } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%arg9, %arg10) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [3584, 128, 1, 1], strides = [1, 1]
        } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<3584x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x3584x256x4xf16, {order = #NHWC}>

        %4 = VPU.NCE.Eltwise(%3, %2) {
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } -> tensor<1x3584x256x4xf16, {order = #NHWC}>

        %5 = VPU.NCE.Eltwise(%4, %1) {
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>
        } -> tensor<1x3584x256x4xf16, {order = #NHWC}>

        VPU.Yield %5
    }

    return %0: tensor<1x3584x256x4xf16, {order = #NHWC}>

    //CHECK:    [[VF:%.+]] = VPU.VerticalFusion (
    //CHECK-SAME:       [[INPUT0]] as [[ARG0:[^:]+]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
    //CHECK-SAME:       %cst as [[ARG1:[^:]+]]: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:       [[INPUT1]] as [[ARG3:[^:]+]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
    //CHECK-SAME:       %cst_0 as [[ARG4:[^:]+]]: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>,
    //CHECK-SAME:       [[INPUT2]] as [[ARG6:[^:]+]]: tensor<1x128x256x4xf16, {order = #NHWC}>,
    //CHECK-SAME:       %cst_1 as [[ARG7:[^:]+]]: tensor<3584x128x1x1x!qElemType, {order = #NHWC}>
    //CHECK-SAME:       ) attributes {tilingStrategy = [1, 1, 3, 1]} -> tensor<1x3584x256x4xf16, {order = #NHWC}> {
    //CHECK:        [[CONV_0:%.+]] = VPU.NCE.Convolution([[ARG6]], [[ARG7]])
    //CHECK:        [[CONV_1:%.+]] = VPU.NCE.Convolution([[ARG3]], [[ARG4]])
    //CHECK:        [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[CONV_0]], [[CONV_1]])
    //CHECK:        [[CONV_2:%.+]] = VPU.NCE.Convolution([[ARG0]], [[ARG1]])
    //CHECK:        [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[ELTWISE_0]], [[CONV_2]])
    //CHECK:        VPU.Yield [[ELTWISE_1]]

    //CHECK:  return [[VF]] : tensor<1x3584x256x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.096478944666245403:128>
!qElemType2 = !quant.uniform<u8:f16, 0.13300120783787148:128>

// CHECK-LABEL: @NotReorderOpsOutsideVFBlock
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<1x64x30x30xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<1x64x30x30xf16, {order = #NHWC}>
func.func @NotReorderOpsOutsideVFBlock(%arg0: tensor<1x64x30x30xf16, {order = #NHWC}>, %arg1: tensor<1x64x30x30xf16, {order = #NHWC}>) -> tensor<1x64x15x15x!qElemType2, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x64x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %0 = VPU.VerticalFusion (%arg0 as %arg2: tensor<1x64x30x30xf16, {order = #NHWC}>, %cst_0 as %arg3: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x30x30x!qElemType1, {order = #NHWC}> {
        %5 = VPU.NCE.DepthConvolution(%arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 1, 1, 1], strides = [1, 1]} -> tensor<1x64x30x30x!quant.uniform<u8:f16, 0.096478944666245403:128>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        VPU.Yield %5
    }
    %1 = VPU.NCE.Convolution(%0, %cst_1) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [32, 64, 3, 3], strides = [2, 2]} : tensor<1x64x30x30x!qElemType1, {order = #NHWC}>, tensor<32x64x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x15x15x!qElemType2, {order = #NHWC}>

    %cst_2 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<2.0> : tensor<64x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x64x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf32>, [#const.CastElemType<f16>, #const.CastElemType<ui8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]
    %2 = VPU.VerticalFusion (%arg1 as %arg2: tensor<1x64x30x30xf16, {order = #NHWC}>, %cst_2 as %arg3: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x30x30x!qElemType1, {order = #NHWC}> {
        %5 = VPU.NCE.DepthConvolution(%arg2, %arg3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [64, 1, 1, 1], strides = [1, 1]} -> tensor<1x64x30x30x!quant.uniform<u8:f16, 0.096478944666245403:128>, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
        VPU.Yield %5
    }
    %3 = VPU.NCE.Convolution(%2, %cst_3) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [32, 64, 3, 3], strides = [2, 2]} : tensor<1x64x30x30x!qElemType1, {order = #NHWC}>, tensor<32x64x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x15x15x!qElemType2, {order = #NHWC}>

    %4 = VPU.Concat(%3, %1) {static_offsets = [[0, 0, 0, 0], [0, 32, 0, 0]]} : tensor<1x32x15x15x!qElemType2, {order = #NHWC}>, tensor<1x32x15x15x!qElemType2, {order = #NHWC}> -> tensor<1x64x15x15x!qElemType2, {order = #NHWC}>

    return %4: tensor<1x64x15x15x!qElemType2, {order = #NHWC}>

    //CHECK:    [[VF_0:%.+]] = VPU.VerticalFusion (
    //CHECK-SAME:       [[INPUT0]] as [[ARG0:[^:]+]]: tensor<1x64x30x30xf16, {order = #NHWC}>,
    //CHECK-SAME:       %cst as [[ARG1:[^:]+]]: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x30x30x!qElemType2, {order = #NHWC}> {
    //CHECK:        [[DW_CONV_0:%.+]] = VPU.NCE.DepthConvolution([[ARG0]], [[ARG1]])
    //CHECK:    [[CONV_0:%.+]] = VPU.NCE.Convolution([[VF_0]], %cst_0)

    //CHECK:    [[VF_1:%.+]] = VPU.VerticalFusion (
    //CHECK-SAME:       [[INPUT1]] as [[ARG3:[^:]+]]: tensor<1x64x30x30xf16, {order = #NHWC}>,
    //CHECK-SAME:       %cst_1 as [[ARG4:[^:]+]]: tensor<64x16x1x1xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 1]} -> tensor<1x64x30x30x!qElemType2, {order = #NHWC}> {
    //CHECK:        [[DW_CONV_1:%.+]] = VPU.NCE.DepthConvolution([[ARG3]], [[ARG4]])
    //CHECK:    [[CONV_1:%.+]] = VPU.NCE.Convolution([[VF_1]], %cst_2)

    //CHECK:    [[CONCAT:%.+]] = VPU.Concat([[CONV_1]], [[CONV_0]])

    //CHECK:    return [[CONCAT]] : tensor<1x64x15x15x!qElemType, {order = #NHWC}>
}
