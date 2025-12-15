//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-dynamic-ops-before-bufferization %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @Unsqueeze {

// CHECK-LABEL:  func.func @UnsqueezeToDynamicReshape
// CHECK-SAME:       ([[ARG:%.+]]: tensor<1x1x10xf16
func.func @UnsqueezeToDynamicReshape(%arg0: tensor<1x1x10xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1]>: tensor<3xsi64>, order = #CHW}>) -> tensor<1x1x10x1xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]>: tensor<4xsi64>, order = #NCHW}> {
    %0 = VPU.Unsqueeze(%arg0) {axes_value = [3]} : tensor<1x1x10xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1]>: tensor<3xsi64>, order = #CHW}> -> tensor<1x1x10x1xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]>: tensor<4xsi64>, order = #NCHW}>
    return %0 : tensor<1x1x10x1xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 1, 0]>: tensor<4xsi64>, order = #NCHW}>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<4xsi32> = dense<[1, 1, 0, 1]> : tensor<4xsi32>
    // CHECK:       VPU.DynamicReshape([[ARG]], [[CST]])
    // CHECK-SAME:      output_bounds = [1, 1, 10, 1]
    // CHECK-SAME:      output_shape = [1, 1, -9223372036854775808, 1]
  }
}
