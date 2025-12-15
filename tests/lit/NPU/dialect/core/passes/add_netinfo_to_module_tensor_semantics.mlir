//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler=vpu-arch=%arch% --split-input-file --add-netinfo-to-module=has-tensor-semantics=true %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


// CHECK-LABEL:  module @Module0
module @Module0 {
    func.func private @nested(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
        return %arg0, %arg1: tensor<f32>, tensor<f32>
    }

    // CHECK:     net.NetworkInfo entryPoint : @nested inputsInfo : {
    // CHECK:         DataInfo "in_0" : tensor<f32>
    // CHECK:         DataInfo "in_1" : tensor<f32>
    // CHECK:     } outputsInfo : {
    // CHECK:         DataInfo "out_0" : tensor<f32>
    // CHECK:         DataInfo "out_1" : tensor<f32>
    // CHECK:     }
    // CHECK:     func.func private @nested
}
