//
// Copyright (C) 2026 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW allow-custom-values=true" --reduce-num-tiles-for-small-models %s | FileCheck %s
// REQUIRES: platform-NPU5010

// CHECK-LABEL: @MatMulMultiplySoftMaxNumClustersReduced
module @MatMulMultiplySoftMaxNumClustersReduced {
    config.Resources 6 of @NCE at 1.850000e+03 MHz {
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }
    config.Resources 1 of @global {
        config.ExecutorResource 2 of @DMA_NN
    }
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1x55x55xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x1x55x128xf16>
    }

    func.func @main(%input: tensor<1x1x55x55xf16>) -> tensor<1x1x55x128xf16> {
        %scale = const.Declare tensor<1x1x55x128xf16> = dense<1.0> : tensor<1x1x55x128xf16>
        %weights = const.Declare tensor<1x1x128x55xf16> = dense<1.0> : tensor<1x1x128x55xf16>
        %matmul = IE.MatMul(%input, %weights) {transpose_b} : tensor<1x1x55x55xf16>, tensor<1x1x128x55xf16> -> tensor<1x1x55x128xf16>
        %multiply = IE.Multiply(%matmul, %scale) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x55x128xf16>, tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        %softmax = IE.SoftMax(%multiply) {axisInd = 3 : i64} : tensor<1x1x55x128xf16> -> tensor<1x1x55x128xf16>
        return %softmax : tensor<1x1x55x128xf16>
    }
}

// CHECK: config.Resources 1 of @NCE
// CHECK: config.ExecutorResource 2 of @DMA_NN

