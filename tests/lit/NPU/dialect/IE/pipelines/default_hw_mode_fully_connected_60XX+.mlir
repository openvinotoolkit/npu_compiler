//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW enable-dcim=true power-optimized=false enable-adaptive-stripping=false" --default-hw-mode-ie --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: platform-NPU6010

// Preserve U16 activation quantization for FullyConnected with U8 WAI,
// while the Multiply sharing the same FakeQuantize/view uses the floating point branch

module @FullyConnectedWithSharedU16FakeQuantize {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x16x32xf32>
        DataInfo "weights" : tensor<64x32xui8>
    } outputsInfo : {
        DataInfo "fc_output" : tensor<16x64xf32>
        DataInfo "multiply_output" : tensor<16x32xf32>
    }

    func.func @main(%input: tensor<1x16x32xf32>, %weights: tensor<64x32xui8>)
            -> (tensor<16x64xf32>, tensor<16x32xf32>) {
        %low = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
        %high = const.Declare tensor<1x1x1xf32> = dense<255.0> : tensor<1x1x1xf32>
        %scale = const.Declare tensor<1x1xf32> = dense<1.0> : tensor<1x1xf32>
        %weightsFloat = IE.DynamicDequantize(%weights, %scale) {dstElemType = f32, vpux.weights_import_dyn_dequant} :
                tensor<64x32xui8>, tensor<1x1xf32> -> tensor<64x32xf32>
        %fq = IE.FakeQuantize(%input, %low, %high, %low, %high)
                {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels_or_low_fp_type = 65536 : i64} :
                tensor<1x16x32xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>,
                tensor<1x1x1xf32>, tensor<1x1x1xf32> -> tensor<1x16x32xf32>
        %view = IE.Reshape(%fq) {shape_value = [16, 32]} : tensor<1x16x32xf32> -> tensor<16x32xf32>
        %fc = IE.FullyConnected(%view, %weightsFloat) :
                tensor<16x32xf32>, tensor<64x32xf32> -> tensor<16x64xf32>
        %multiply = IE.Multiply(%view, %view) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                tensor<16x32xf32>, tensor<16x32xf32> -> tensor<16x32xf32>
        return %fc, %multiply : tensor<16x64xf32>, tensor<16x32xf32>

        // CHECK-DAG: !qElemType = !quant.uniform<u8:f16, 1.000000e+00>
        // CHECK-DAG: !qElemType1 = !quant.uniform<u16:f16, 0.0038910505836575876>
        // CHECK-LABEL: @FullyConnectedWithSharedU16FakeQuantize
        // CHECK: func.func @main(
        // CHECK-SAME: [[INPUT:%[^:]+]]: tensor<1x16x32xf32>, [[WEIGHTS:%[^:]+]]: tensor<64x32xui8>
        // CHECK: [[INPUT_4D:%.+]] = IE.AffineReshape([[INPUT]])
        // CHECK: [[WEIGHTS_Q:%.+]] = IE.QuantizeCast([[WEIGHTS]]) {dstElemType = !qElemType}
        // CHECK: [[INPUT_F16:%.+]] = IE.Convert([[INPUT_4D]]) {dstElemType = f16}
        // CHECK: [[INPUT_LAYOUT:%.+]] = IE.LayoutCast([[INPUT_F16]])
        // CHECK: [[INPUT_SHAPE:%.+]] = IE.ShapeCast {{.*}} inputs([[INPUT_LAYOUT]] :
        // CHECK: [[QUANTIZED:%.+]] = VPU.NCE.DepthConvolution([[INPUT_SHAPE]],
        // CHECK-SAME: -> tensor<1x16x16x2x!qElemType1, {order = #NHWC}>
        // CHECK: [[QUANTIZED_SHAPE:%.+]] = IE.ShapeCast {{.*}} inputs([[QUANTIZED]] :
        // CHECK: [[QUANTIZED_LAYOUT:%.+]] = IE.LayoutCast([[QUANTIZED_SHAPE]])
        // CHECK: [[FLOAT_LAYOUT:%.+]] = IE.PermuteCast([[INPUT_F16]])
        // CHECK: [[FLOAT_SHAPE:%.+]] = IE.ShapeCast {{.*}} inputs([[FLOAT_LAYOUT]] :
        // CHECK: [[RELU:%.+]] = VPU.NCE.AveragePool([[FLOAT_SHAPE]])
        // CHECK-SAME: ppe = #VPU.PPEFp<mode = <LRELU>, clamp_low = 0.000000e+00 : f64,
        // CHECK-SAME: prelu_alpha = [-0.000000e+00]
        // CHECK-SAME: -> tensor<1x16x8x4xf16, {order = #NHWC}>
        // CHECK: [[RELU_SHAPE:%.+]] = IE.ShapeCast {{.*}} inputs([[RELU]] :
        // CHECK: [[RELU_LAYOUT:%.+]] = IE.PermuteCast([[RELU_SHAPE]])
        // CHECK: [[FC_INPUT:%.+]] = IE.AffineReshape([[QUANTIZED_LAYOUT]])
        // CHECK: [[FC_WEIGHTS:%.+]] = IE.AffineReshape([[WEIGHTS_Q]])
        // CHECK: [[FC_INPUT_LAYOUT:%.+]] = IE.PermuteCast([[FC_INPUT]])
        // CHECK: [[FC_WEIGHTS_LAYOUT:%.+]] = IE.PermuteCast([[FC_WEIGHTS]])
        // CHECK: [[FC_INPUT_SHAPE:%.+]] = IE.AffineReshape([[FC_INPUT_LAYOUT]])
        // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[FC_INPUT_SHAPE]], [[FC_WEIGHTS_LAYOUT]],
        // CHECK-SAME: rawFilterShape [64, 32, 1, 1]
        // CHECK-SAME: mpe_engine = #VPU.MPEEngine60XX<mode = <DCIM>, weight_zp = 0 : i64, activation_zp = 0 : i64>
        // CHECK-SAME: : tensor<1x32x4x4x!qElemType1, {order = #NHWC}>, tensor<64x32x1x1x!qElemType, {order = #NHWC}>
        // CHECK-SAME: -> tensor<1x64x4x4xf16, {order = #NHWC}>
        // CHECK: [[MULTIPLY_INPUT:%.+]] = IE.AffineReshape([[RELU_LAYOUT]])
        // CHECK: [[MULTIPLY_LAYOUT:%.+]] = IE.PermuteCast([[MULTIPLY_INPUT]])
        // CHECK: [[MULTIPLY:%.+]] = VPU.NCE.Eltwise([[MULTIPLY_LAYOUT]], [[MULTIPLY_LAYOUT]])
        // CHECK-SAME: op_type = #VPU.eltwise_type<MULTIPLY>
        // CHECK-SAME: -> tensor<1x512x1x1xf16, {order = #NHWC}>
        // CHECK: [[MULTIPLY_OUTPUT_LAYOUT:%.+]] = IE.PermuteCast([[MULTIPLY]])
        // CHECK: [[MULTIPLY_OUTPUT_SHAPE:%.+]] = IE.Reshape([[MULTIPLY_OUTPUT_LAYOUT]])
        // CHECK: [[FC_OUTPUT_F32:%.+]] = VPU.NCE.AveragePool([[CONV]])
        // CHECK-SAME: -> tensor<1x64x4x4xf32, {order = #NHWC}>
        // CHECK: [[FC_OUTPUT_SHAPE:%.+]] = IE.AffineReshape([[FC_OUTPUT_F32]])
        // CHECK: [[FC_OUTPUT_LAYOUT:%.+]] = IE.PermuteCast([[FC_OUTPUT_SHAPE]])
        // CHECK: [[FC_OUTPUT:%.+]] = IE.AffineReshape([[FC_OUTPUT_LAYOUT]])
        // CHECK: [[MULTIPLY_OUTPUT_F32:%.+]] = IE.Convert([[MULTIPLY_OUTPUT_SHAPE]]) {dstElemType = f32}
        // CHECK: [[MULTIPLY_OUTPUT:%.+]] = IE.AffineReshape([[MULTIPLY_OUTPUT_F32]])
        // CHECK-NEXT: return [[FC_OUTPUT]], [[MULTIPLY_OUTPUT]] : tensor<16x64xf32>, tensor<16x32xf32>
    }
}
