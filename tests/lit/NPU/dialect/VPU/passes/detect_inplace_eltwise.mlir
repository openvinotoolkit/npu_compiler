//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --detect-in-place-eltwise %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @inplaceFp16Eltwise (%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                          %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                          -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %2 : tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-SAME:           is_inplace = true,
    //CHECK-SAME:           op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>
    //CHECK-SAME:   } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.011588541666666667:128>
!qElemType1 = !quant.uniform<u8:f16, 0.020557598039215686:128>
!qElemType2 = !quant.uniform<u8:f16, 0.0088848039215686271:128>

func.func @inplaceQuantEltwise (%arg0: tensor<1x256x56x56x!qElemType, {order = #NHWC}>,
                           %arg1: tensor<1x256x56x56x!qElemType1, {order = #NHWC}>)
                           -> tensor<1x256x56x56x!qElemType2, {order = #NHWC}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56x!qElemType, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x256x56x56x!qElemType1, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x256x56x56x!qElemType2, {order = #NHWC}>

    return %2 : tensor<1x256x56x56x!qElemType2, {order = #NHWC}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-SAME:           is_inplace = true,
    //CHECK-SAME:           op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>
    //CHECK-SAME:   } -> tensor<1x256x56x56x!qElemType2, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @fp16EltwiseBlockArg (%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                           %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                           -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    //CHECK-NOT:           is_inplace = true,
    //CHECK-SAME:           op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>
    //CHECK-SAME:   } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        return [[ELTWISE]] : tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @Fp16EltwiseFitIntoCMX (%arg0: tensor<1x16x56x56xf16, {order = #NHWC}>,
                             %arg1: tensor<1x16x56x56xf16, {order = #NHWC}>)
                             -> tensor<1x16x56x56xf16, {order = #NHWC}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    return %2 : tensor<1x16x56x56xf16, {order = #NHWC}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-NOT:            is_inplace = true,
    //CHECK-SAME:           op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>
    //CHECK-SAME:   } -> tensor<1x16x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @Fp16EltwiseDiffLayout (%arg0: tensor<1x16x56x56xf16, {order = #NHWC}>,
                             %arg1: tensor<1x16x56x56xf16, {order = #NHWC}>)
                             -> tensor<1x16x56x56xf16, {order = #NCHW}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x56x56xf16, {order = #NCHW}>

    return %2 : tensor<1x16x56x56xf16, {order = #NCHW}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-NOT:            is_inplace = true,
    //CHECK-SAME:           op_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:           ppe = #VPU.PPEStub<>
    //CHECK-SAME:   } -> tensor<1x16x56x56xf16, {order = #NCHW}>
}
