//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=ReferenceSW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --reference-sw-mode-vpu %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: @SoftMax
module @SoftMax {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        %1 = VPU.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        return %2 : tensor<1x1000xf16>

        // CHECK:               [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[COPY_CMX:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = [@CMX_NN, 0]}
        // CHECK-SAME:            : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        // CHECK:               [[SOFTMAX:%.+]] = VPU.SoftMax([[COPY_CMX]]) {axisInd = 3 : i64}
        // CHECK-SAME:            : tensor<1x1x1x1000xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1x1000xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        // CHECK:               [[COPY_DDR:%.+]] = VPU.Copy([[SOFTMAX]])
        // CHECK-SAME:            : tensor<1x1x1x1000xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[OUT:%.+]] = VPU.AffineReshape([[COPY_DDR]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
    }
}
