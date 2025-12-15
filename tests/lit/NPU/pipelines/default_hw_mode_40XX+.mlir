//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// TODO: #185311 remove enable-reorder-concat-branches option
// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --default-hw-mode-ie="verify-locations=off" --lower-IE-to-VPU --default-hw-mode-vpu="concat-repeating-block-outlining-min-seq-length=1 vf-outlining=false enable-reorder-concat-branches=false" %s -o %t
// RUN: FileCheck --check-prefix=CHECK-VPU %s --input-file %t

// "init-compiler" is used here instead of "vpu-arch" only because of the error
// OneShotBufferizeVPU2VPUIP failed: "ppe_version_config.cpp Tried to access an uninitialized PpeFactory"
// Looks like this is "by design"
// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --lower-VPU-to-VPUIP --default-hw-mode-vpuip="vf-outlining=false" %t | FileCheck --check-prefix=CHECK-VPUIP %s

// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @OutlineConcat {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x48x32x32xf16>
        DataInfo "input1" : tensor<1x48x32x32xf16>
        DataInfo "input2" : tensor<1x96x32x32xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x192x32x32xf16>
    }

    func.func @main(%input0: tensor<1x48x32x32xf16>, %input1: tensor<1x48x32x32xf16>, %input2: tensor<1x96x32x32xf16>) -> tensor<1x192x32x32xf16> {
        %softmax1 = IE.SoftMax(%input0) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %relu1 = IE.ReLU(%softmax1) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %softmax2 = IE.SoftMax(%input1) {axisInd = 1 : i64} : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>
        %relu2 = IE.ReLU(%softmax2) : tensor<1x48x32x32xf16> -> tensor<1x48x32x32xf16>

        %concat1 = IE.Concat(%relu1, %relu2) {static_offsets = [[0, 0, 0, 0], [0, 48, 0, 0]]}
            : tensor<1x48x32x32xf16>, tensor<1x48x32x32xf16> -> tensor<1x96x32x32xf16>

        %softmax3 = IE.SoftMax(%concat1) {axisInd = 1 : i64} : tensor<1x96x32x32xf16> -> tensor<1x96x32x32xf16>
        %relu3 = IE.ReLU(%softmax3) : tensor<1x96x32x32xf16> -> tensor<1x96x32x32xf16>
        %softmax4 = IE.SoftMax(%input2) {axisInd = 2 : i64} : tensor<1x96x32x32xf16> -> tensor<1x96x32x32xf16>
        %relu4 = IE.ReLU(%softmax4) : tensor<1x96x32x32xf16> -> tensor<1x96x32x32xf16>

        %concat2 = IE.Concat(%relu3, %relu4) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]}
            : tensor<1x96x32x32xf16>, tensor<1x96x32x32xf16> -> tensor<1x192x32x32xf16>

        return %concat2 : tensor<1x192x32x32xf16>
    }

    // Make sure main function contains nested function calls
    // CHECK-VPU: func.func @main([[ARG0:%.+]]: tensor<1x48x32x32xf16>, [[ARG1:%.+]]: tensor<1x48x32x32xf16>, [[ARG2:%.+]]: tensor<1x96x32x32xf16>)
    // CHECK-VPU:     [[CONCAT1:%.+]] = call @main_concat1([[ARG0]], [[ARG1]])
    // CHECK-VPU:     [[CONCAT2:%.+]] = call @main_concat2([[CONCAT1]], [[ARG2]])
    // CHECK-VPU:     return [[CONCAT2]] : tensor<1x192x32x32xf16>

    // Original bug was that when vf-outlining is disabled, the function calls were not inlined
    // Check that after VPUIP pipeline is finished there are no function calls left
    // CHECK-VPUIP:        func.func @main([[ARG0:%.+]]: memref<1x48x32x32xf16, @DDR>, [[ARG1:%.+]]: memref<1x48x32x32xf16, @DDR>, [[ARG2:%.+]]: memref<1x96x32x32xf16, @DDR>, [[OUT:%.+]]: memref<1x192x32x32xf16, @DDR>)
    // CHECK-VPUIP-NOT:        func.call
}
