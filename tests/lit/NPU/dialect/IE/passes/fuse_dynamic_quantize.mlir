//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --fuse-dynamic-quantize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @FuseDQ
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x304x560xf32>
func.func @FuseDQ(%arg0: tensor<1x304x560xf32>) -> (tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<255.> : tensor<1x1x1xf32> isSplat
    %cst_0 = const.Declare tensor<1xf32> = dense<0.0039215> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %cst_1 = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32> isSplat, [#const.Reshape<[1]>]
    %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2]} : tensor<1x304x560xf32> -> tensor<1xf32>
    %1 = IE.Clamp(%0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Subtract(%cst_1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %3 = IE.ReduceMax(%arg0) {axes_value = [0, 1, 2]} : tensor<1x304x560xf32> -> tensor<1xf32>
    %4 = IE.Clamp(%3) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1xf32> -> tensor<1xf32>
    %5 = IE.Subtract(%4, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %6 = IE.Multiply(%5, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %7 = IE.Divide(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %8 = IE.Round(%7) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1xf32> -> tensor<1xf32>
    %9 = IE.Clamp(%8) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1xf32> -> tensor<1xf32>
    %10 = IE.Convert(%9) {dstElemType = ui8} : tensor<1xf32> -> tensor<1xui8>
    %11 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1x1x1xf32> -> tensor<1x304x560xf32>
    %12 = IE.Divide(%11, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1xf32> -> tensor<1x304x560xf32>
    %13 = IE.Round(%12) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x304x560xf32> -> tensor<1x304x560xf32>
    %14 = IE.Add(%13, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1xf32> -> tensor<1x304x560xf32>
    %15 = IE.Clamp(%14) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x304x560xf32> -> tensor<1x304x560xf32>
    %16 = IE.Convert(%15) {dstElemType = ui8} : tensor<1x304x560xf32> -> tensor<1x304x560xui8>
    return %16, %6, %10 : tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Clamp
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add

    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ReduceMin]], [[ReduceMax]])
    // CHECK-SAME: -> tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>
    // CHECK: return [[OUT]], [[SCALE]], [[ZP]] : tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>
}
