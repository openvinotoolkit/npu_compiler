//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW" --mlir-elide-elementsattrs-if-larger 8 --reference-sw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {

    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>
    }

    // CHECK:       [[CST:%.+]] = const.Declare tensor<48x3x3x3xf16> = dense<1.000000e+00> :
    // CHECK-SAME:                      tensor<48x3x3x3xf32>, [#const.CastElemType<f16>]
    // CHECK:       [[OUT:%.+]] = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME: tensor<1x3x62x62xf16>, tensor<48x3x3x3xf16> -> tensor<1x48x60x60xf16>
    // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>

}


// -----

module @SoftMax {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x24x64xf16>
    }
    outputsInfo : {
        DataInfo "grn" : tensor<1x8x24x64xf16>
    }

    func.func @main(%arg0: tensor<1x8x24x64xf16>) -> tensor<1x8x24x64xf16> {
        %0 = IE.GRN(%arg0) {bias = 0.33000001311302185 : f64} : tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>
        return %0 : tensor<1x8x24x64xf16>
    }

    // CHECK:       [[NORMALIZEL2:%.*]] = IE.NormalizeL2(%arg0)
    // CHECK-SAME:       {axes_value = [1 : si64], eps = 0.33000001311302185 : f64, eps_mode = #IE.eps_mode<ADD>}
    // CHECK-SAME:        : tensor<1x8x24x64xf16> -> tensor<1x8x24x64xf16>

    // CHECK:       return [[NORMALIZEL2]]
}
