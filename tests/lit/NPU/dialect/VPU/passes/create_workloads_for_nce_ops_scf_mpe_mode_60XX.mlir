//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --create-workloads-for-nce-ops-scf %s | FileCheck %s
// REQUIRES: platform-NPU6010

// Verify engine-specific MPE mode selection for convolutions and grouped matmuls.

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @ConvolutionSelectsDCIMIC128
func.func @ConvolutionSelectsDCIMIC128(
        %input: tensor<1x256x16x12x!qElemType, {order = #NHWC}>,
        %weights: tensor<256x256x1x1x!qElemType, {order = #NHWC}>)
        -> tensor<1x256x16x12x!qElemType, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%out = %input) -> (tensor<1x256x16x12x!qElemType, {order = #NHWC}>) {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [256, 256, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 128 : i64, activation_zp = 128 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<1x256x16x12x!qElemType, {order = #NHWC}>, tensor<256x256x1x1x!qElemType, {order = #NHWC}> -> tensor<1x256x16x12x!qElemType, {order = #NHWC}>
        scf.yield %0 : tensor<1x256x16x12x!qElemType, {order = #NHWC}>
    }
    return %result : tensor<1x256x16x12x!qElemType, {order = #NHWC}>

    // CHECK:       scf.for
    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:  minimumHardwareExecutionCost = {{[1-9][0-9]{0,5} }}: i64
    // CHECK-SAME:  mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>
    // CHECK:       VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 256, 16, 12] outOffsets [0, 0, 0, 0] outSizes [1, 256, 16, 12] pad [0, 0, 0, 0] <DCIM_IC128>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @ConvolutionUsesCuboidHeuristicForSCL
func.func @ConvolutionUsesCuboidHeuristicForSCL(
        %input: tensor<1x256x16x12x!qElemType, {order = #NHWC}>,
        %weights: tensor<256x256x1x1x!qElemType, {order = #NHWC}>)
        -> tensor<1x256x16x12x!qElemType, {order = #NHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%out = %input) -> (tensor<1x256x16x12x!qElemType, {order = #NHWC}>) {
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [256, 256, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 128 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<1x256x16x12x!qElemType, {order = #NHWC}>, tensor<256x256x1x1x!qElemType, {order = #NHWC}> -> tensor<1x256x16x12x!qElemType, {order = #NHWC}>
        scf.yield %0 : tensor<1x256x16x12x!qElemType, {order = #NHWC}>
    }
    return %result : tensor<1x256x16x12x!qElemType, {order = #NHWC}>

    // CHECK:       scf.for
    // CHECK:       VPU.NCE.Convolution
    // CHECK-SAME:  minimumHardwareExecutionCost = {{[1-9][0-9]{0,5} }}: i64
    // CHECK-SAME:  mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>
    // CHECK:       VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 256, 16, 12] outOffsets [0, 0, 0, 0] outSizes [1, 256, 16, 12] pad [0, 0, 0, 0] <CUBOID_8x16>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @GroupedMatMulPreservesDCIMIC128
func.func @GroupedMatMulPreservesDCIMIC128(
        %input: tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>,
        %weights: tensor<2x256x256x1x1x!qElemType, {order = #GNHWC}>)
        -> tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%out = %input) -> (tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>) {
        %0 = VPU.NCE.MatMul(%input, %weights) rawFilterShape [2, 256, 256, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 128 : i64, activation_zp = 128 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>, tensor<2x256x256x1x1x!qElemType, {order = #GNHWC}> -> tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>
        scf.yield %0 : tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>
    }
    return %result : tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>

    // CHECK:       VPU.NCE.MatMul
    // CHECK-SAME:  mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>
    // CHECK:       VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [2, 1, 256, 16, 1] outOffsets [0, 0, 0, 0, 0] outSizes [2, 1, 256, 16, 1] pad [0, 0, 0, 0] <DCIM_IC128>
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00:128>

// CHECK-LABEL: @GroupedMatMulPreservesCuboidFallbackForSCL
func.func @GroupedMatMulPreservesCuboidFallbackForSCL(
        %input: tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>,
        %weights: tensor<2x256x256x1x1x!qElemType, {order = #GNHWC}>)
        -> tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %c1 step %c1 iter_args(%out = %input) -> (tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>) {
        %0 = VPU.NCE.MatMul(%input, %weights) rawFilterShape [2, 256, 256, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>, weight_zp = 128 : i64, activation_zp = 128 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } : tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>, tensor<2x256x256x1x1x!qElemType, {order = #GNHWC}> -> tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>
        scf.yield %0 : tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>
    }
    return %result : tensor<2x1x256x16x1x!qElemType, {order = #GNHWC}>

    // CHECK:       VPU.NCE.MatMul
    // CHECK-SAME:  mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>
    // CHECK:       VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [2, 1, 256, 16, 1] outOffsets [0, 0, 0, 0, 0] outSizes [2, 1, 256, 16, 1] pad [0, 0, 0, 0] <CUBOID_16x16>
}
