//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Verifies that the legacy ConvolutionSplitOverInputChannel pass is scheduled
// in the NPU5020 default-hw-mode pipeline.

// RUN: vpux-opt --platform=%platform% --mlir-elide-elementsattrs-if-larger=8 --mlir-print-ir-after=convolution-split-over-input-channel --default-hw-mode %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU5020

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @WCLLegacySplitOverIC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2048x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x256x32x32xf16>
    }

    func.func @main(%arg0: tensor<1x2048x32x32xf16>) -> tensor<1x256x32x32xf16> {
        %weights = const.Declare tensor<256x2048x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x2048x3x3xf16, {order = #NHWC}>
        %conv = IE.Convolution(%arg0, %weights) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x2048x32x32xf16>, tensor<256x2048x3x3xf16, {order = #NHWC}> -> tensor<1x256x32x32xf16>
        return %conv : tensor<1x256x32x32xf16>
    }
}

// CHECK:      IR Dump After ConvolutionSplitOverInputChannel
// CHECK-DAG:  VPU.Slice {{.+}} [0, 0, 0, 0] [1, [[IC_TILE:[0-9]+]], 32, 32]
// CHECK-DAG:  VPU.Slice {{.+}} [0, [[IC_TILE]], 0, 0] [1, [[IC_TILE]], 32, 32]
// CHECK:      VPU.NCE.Convolution({{.+}}) rawFilterShape [256, [[IC_TILE]], 3, 3]
// CHECK:      VPU.NCE.Eltwise
// CHECK-SAME: op_type = #VPU.eltwise_type<ADD>
