//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-grouped-matmul=false convert-cumsum-to-matmul=true" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// The functional-shape case is intentionally kept here to make pipeline behavior explicit.
// CHECK-LABEL: @CumSumFunctionalCaseNoFuse
module @CumSumFunctionalCaseNoFuse {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<5x14x5x7xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<5x14x5x7xf16>
    }

    // CHECK-LABEL: func.func @main
    // CHECK:       IE.CumSum
    // CHECK-NOT:   IE.MatMul
    func.func @main(%arg0: tensor<5x14x5x7xf16>) -> tensor<5x14x5x7xf16> {
        %0 = IE.CumSum(%arg0) {axis_value = 0 : i64, exclusive} : tensor<5x14x5x7xf16> -> tensor<5x14x5x7xf16>
        return %0 : tensor<5x14x5x7xf16>
    }
}

// -----

// For an eligible CumSum, the full DefaultHW IE pipeline first rewrites to MatMul
// and then further lowers MatMul to a Convolution-based form.
// CHECK-LABEL: @CumSum2DConvertedInPipeline
module @CumSum2DConvertedInPipeline {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<256x128xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<256x128xf16>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-NOT:   IE.CumSum
    // CHECK:       const.Declare tensor<256x256x1x1xf16, {order = #NHWC}>
    // CHECK:       IE.Convolution
    func.func @main(%arg0: tensor<256x128xf16>) -> tensor<256x128xf16> {
        %0 = IE.CumSum(%arg0) {axis_value = 0 : i64} : tensor<256x128xf16> -> tensor<256x128xf16>
        return %0 : tensor<256x128xf16>
    }
}

// -----

// Exact functional-test shape: {1, 64, 4, 256, 256}, axis=3.
// CHECK-LABEL: @CumSum5D_1x64x4x256x256_ConvertedInPipeline
module @CumSum5D_1x64x4x256x256_ConvertedInPipeline {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x4x256x256xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x64x4x256x256xf16>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-NOT:   IE.CumSum
    // CHECK:       const.Declare tensor<256x256x1x1xf16, {order = #NHWC}>
    // CHECK:       IE.Convolution
    func.func @main(%arg0: tensor<1x64x4x256x256xf16>) -> tensor<1x64x4x256x256xf16> {
        %0 = IE.CumSum(%arg0) {axis_value = 3 : i64} : tensor<1x64x4x256x256xf16> -> tensor<1x64x4x256x256xf16>
        return %0 : tensor<1x64x4x256x256xf16>
    }
}
