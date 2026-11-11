//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --reference-sw-mode %s | FileCheck %s -DPLATFORM=%platform%
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU5020 || platform-NPU6010

// CHECK:       module @SoftMax
// CHECK-SAME:      config.compilationMode = #config.compilation_mode<ReferenceSW>
// CHECK-SAME:      config.platform = #config.platform<[[PLATFORM]]>
module @SoftMax {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK:       func.func @main
    // CHECK-SAME:      ({{%[^:]+}}: memref<1x1000xf16, @DDR>, [[ARG1:%[^:]+]]: memref<1x1000xf16, @DDR>) -> memref<1x1000xf16, @DDR>
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf16> -> tensor<1x1000xf16>
        return %0 : tensor<1x1000xf16>
    }

    // CHECK-NOT:   VPUIP.NCEClusterTask
    // CHECK:       VPUIP.SW.Kernel {resultSegmentSizes = array<i32: 1, 0, 0>} @VPU.SW::@builtin_SoftMax
    // CHECK:       return [[ARG1]] : memref<1x1000xf16, @DDR>
}
