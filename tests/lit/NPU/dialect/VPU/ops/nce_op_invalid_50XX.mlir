//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt %s --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConvMaxKernelSize
func.func @NceConvMaxKernelSize(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
                   -> tensor<1x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    // expected-error@+1 {{Unsupported kernel height dimension '16', must be in range [1, 15]}}
    %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2) rawFilterShape [16, 16, 16, 16] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}> -> tensor<1x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @NceConvInvalidFilterShape {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }
    func.func @main(%input: tensor<1x3x40x80xf16, {order = #NHWC}>,
                    %weights: tensor<16x3x4x8xf16, {order = #NHWC}>)
            -> tensor<1x16x37x73xf16, {order = #NHWC}> {
        // expected-error@+1 {{Got wrong shape for NCE Convolution 'filter' '[16, 3, 4, 8]', expected '[16, 1, 1, 512]'}}
        %0 = VPU.NCE.Convolution(%input, %weights) rawFilterShape [16, 3, 4, 8] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

            strides = [1, 1]
        } : tensor<1x3x40x80xf16, {order = #NHWC}>, tensor<16x3x4x8xf16, {order = #NHWC}> -> tensor<1x16x37x73xf16, {order = #NHWC}>

        return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceDepthConvMaxKernelSize
func.func @NceDepthConvMaxKernelSize(%arg0: tensor<1x16x40x40xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                        %arg1: tensor<16x1x16x16xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x16x25x25xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    // expected-error@+1 {{Unsupported kernel height dimension '16', must be in range [1, 15]}}
    %0 = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [16, 1, 16, 16] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,

                strides = [1, 1]
            } -> tensor<1x16x25x25xf16, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 25, 25] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x25x25xf16, {order = #NHWC, mem_space = @CMX_NN}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceMaxPoolMaxKernelSize
func.func @NceMaxPoolMaxKernelSize(%arg0: tensor<1x16x20x20xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                      %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x16x5x5xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    // expected-error@+1 {{Unsupported kernel height dimension '16', must be in range [1, 15]}}
    %0 = VPU.NCE.MaxPool(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                kernel_size = [16, 16],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                strides = [1, 1]
            } -> tensor<1x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 20, 20] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x5x5xf16, {order = #NHWC, mem_space = @CMX_NN}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceAveragePoolMaxKernelSize
func.func @NceAveragePoolMaxKernelSize(%arg0: tensor<1x16x20x20xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                      %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x16x5x5xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    // expected-error@+1 {{Unsupported kernel height dimension '16', must be in range [1, 15]}}
    %0 = VPU.NCE.AveragePool(%arg0) {
            kernel_size = [16, 16],
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1]
        } -> tensor<1x16x5x5xf16, {mem_space = @CMX_NN, order = #NHWC}>

    return %0 : tensor<1x16x5x5xf16, {order = #NHWC, mem_space = @CMX_NN}>
}
