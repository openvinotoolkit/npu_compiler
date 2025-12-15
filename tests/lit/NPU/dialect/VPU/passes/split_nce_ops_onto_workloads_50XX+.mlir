//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --split-NCE-ops-onto-workloads %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @SplitNCEPermute
func.func @SplitNCEPermute(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       VPU.NCE.Permute(%arg0) {
    // CHECK-SAME:      dstElemType = !qElemType, dstOrder = #NHWC, expandedChannels = 4 : i64,
    // CHECK-SAME:      minimumHardwareExecutionCost = {{[1-9][0-9]+}} : i64,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
    // CHECK-SAME:      } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {

    // CHECK:       DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 4, 224, 224]
    // CHECK-SAME:      <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEConvWithUnpaddedInputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }
    func.func @main(%input: tensor<1x3x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<16x1x1x512xf16, {order = #NHWC}>)
            -> tensor<1x16x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) {
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
            rawFilterShape = [16, 3, 4, 8],
            strides = [1, 1]
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<16x1x1x512xf16, {order = #NHWC}> -> tensor<1x16x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 37, 73] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEConvWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x16x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<3x16x4x8xf16, {order = #NHWC}>)
            -> tensor<1x3x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) {
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
            rawFilterShape = [3, 16, 4, 8],
            strides = [1, 1]
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<3x16x4x8xf16, {order = #NHWC}> -> tensor<1x3x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 37, 73] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEConvWithUnpaddedInputOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x3x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<3x1x1x512xf16, {order = #NHWC}>)
            -> tensor<1x3x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights) {
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
            rawFilterShape = [3, 3, 4, 8],
            strides = [1, 1]
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<3x1x1x512xf16, {order = #NHWC}> -> tensor<1x3x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.Convolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 37, 73] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEDepthConvWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%input: tensor<1x16x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<3x1x4x8xf16, {order = #NHWC}>)
            -> tensor<1x3x37x73xf16, {order = #NHWC}> {
        %0 = VPU.NCE.DepthConvolution(%input, %weights) {
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
            rawFilterShape = [3, 1, 4, 8],
            strides = [1, 1]
        } -> tensor<1x3x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x3x37x73xf16, {order = #NHWC}>
    }
    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 37, 73] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEMaxPoolWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x3x16x16xf16, {order = #NHWC}>
        return %0 : tensor<1x3x16x16xf16, {order = #NHWC}>
    }

    // CHECK:       VPU.NCE.MaxPool
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEAvgPoolWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {
            input_padding = [0, 13, 0, 0],
            kernel_size = [1, 1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } -> tensor<1x3x16x16xf16, {order = #NHWC}>
        return %0 : tensor<1x3x16x16xf16, {order = #NHWC}>
    }

    // CHECK:       VPU.NCE.AveragePool
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @SplitNCEEltwiseWithUnpaddedOutputChannels {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingODU : true
    }
    func.func @main(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x3x16x16xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
            input_padding = [0, 13, 0, 0],
            op_type = #VPU.eltwise_type<SUBTRACT>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        } -> tensor<1x3x16x16xf16, {order = #NHWC}>
        return %0 : tensor<1x3x16x16xf16, {order = #NHWC}>
    }

    // CHECK:       VPU.NCE.Eltwise
    // CHECK-NEXT:    VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 16, 16] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16>
}
