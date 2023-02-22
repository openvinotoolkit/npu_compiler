//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --detect-in-place-eltwise %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @inplaceFp16Eltwise (%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                          %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                          -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [0.020408163265306121]},
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [0.020408163265306121]},
        strides = [1, 1]
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %2 : tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-SAME:           is_inplace = true,
    //CHECK-SAME:           op_type = "ADD",
    //CHECK-SAME:           ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    //CHECK-SAME:   } -> tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = type !quant.uniform<u8:f16, 0.011588541666666667:128>
!qElemType1 = type !quant.uniform<u8:f16, 0.020557598039215686:128>
!qElemType2 = type !quant.uniform<u8:f16, 0.0088848039215686271:128>

func @inplaceQuantEltwise (%arg0: tensor<1x256x56x56x!qElemType0, {order = #NHWC}>,
                           %arg1: tensor<1x256x56x56x!qElemType1, {order = #NHWC}>)
                           -> tensor<1x256x56x56x!qElemType2, {order = #NHWC}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
        strides = [1, 1]
    } -> tensor<1x256x56x56x!qElemType0, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP"},
        strides = [1, 1]
    } -> tensor<1x256x56x56x!qElemType1, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = "ADD",
        ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24302], in2_quant_mult = [43112], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELUX", quant_mult = [28813], quant_post_shift = 0 : i64, quant_shift = [29]}
    } -> tensor<1x256x56x56x!qElemType2, {order = #NHWC}>

    return %2 : tensor<1x256x56x56x!qElemType2, {order = #NHWC}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-SAME:           is_inplace = true,
    //CHECK-SAME:           op_type = "ADD",
    //CHECK-SAME:           ppe = {clamp_high = 255 : i64, clamp_low = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64, in1_quant_mult = [24302], in2_quant_mult = [43112], lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELUX", quant_mult = [28813], quant_post_shift = 0 : i64, quant_shift = [29]}
    //CHECK-SAME:   } -> tensor<1x256x56x56x!qElemType2, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @fp16EltwiseBlockArg (%arg0: tensor<1x256x56x56xf16, {order = #NHWC}>,
                           %arg1: tensor<1x256x56x56xf16, {order = #NHWC}>)
                           -> tensor<1x256x56x56xf16, {order = #NHWC}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    return %0 : tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise(%arg0, %arg1) {
    //CHECK-NOT:           is_inplace = true,
    //CHECK-SAME:           op_type = "ADD",
    //CHECK-SAME:           ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    //CHECK-SAME:   } -> tensor<1x256x56x56xf16, {order = #NHWC}>

    //CHECK:        return [[ELTWISE]] : tensor<1x256x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func @Fp16EltwiseFitIntoCMX (%arg0: tensor<1x16x56x56xf16, {order = #NHWC}>,
                             %arg1: tensor<1x16x56x56xf16, {order = #NHWC}>)
                             -> tensor<1x16x56x56xf16, {order = #NHWC}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [0.020408163265306121]},
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [0.020408163265306121]},
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    return %2 : tensor<1x16x56x56xf16, {order = #NHWC}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-NOT:            is_inplace = true,
    //CHECK-SAME:           op_type = "ADD",
    //CHECK-SAME:           ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    //CHECK-SAME:   } -> tensor<1x16x56x56xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func @Fp16EltwiseDiffLayout (%arg0: tensor<1x16x56x56xf16, {order = #NHWC}>,
                             %arg1: tensor<1x16x56x56xf16, {order = #NHWC}>)
                             -> tensor<1x16x56x56xf16, {order = #NCHW}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [0.020408163265306121]},
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %1 = VPU.NCE.AveragePool(%arg1) {
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [0.020408163265306121]},
        strides = [1, 1]
    } -> tensor<1x16x56x56xf16, {order = #NHWC}>

    %2 = VPU.NCE.Eltwise(%0, %1) {
        op_type = "ADD",
        ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    } -> tensor<1x16x56x56xf16, {order = #NCHW}>

    return %2 : tensor<1x16x56x56xf16, {order = #NCHW}>

    //CHECK:        [[AVGPOOL0:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[AVGPOOL1:%.*]] = VPU.NCE.AveragePool
    //CHECK:        [[ELTWISE:%.*]] = VPU.NCE.Eltwise([[AVGPOOL0]], [[AVGPOOL1]]) {
    //CHECK-NOT:            is_inplace = true,
    //CHECK-SAME:           op_type = "ADD",
    //CHECK-SAME:           ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, fp_prelu_alpha = 1.000000e+00 : f64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "LRELU", quant_scale = [1.000000e+00]}
    //CHECK-SAME:   } -> tensor<1x16x56x56xf16, {order = #NCHW}>
}
