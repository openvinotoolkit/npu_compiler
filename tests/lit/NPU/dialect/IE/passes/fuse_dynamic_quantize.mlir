//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --fuse-dynamic-quantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @FuseDQ
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x304x560xf32>
func.func @FuseDQ(%arg0: tensor<1x304x560xf32>) -> (tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<255.> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<0.0039215> : tensor<f32>, [#const.Reshape<[1]>]
    %cst_1 = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32>, [#const.Reshape<[1]>]
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
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add

    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]])
    // CHECK-SAME: -> tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>
    // CHECK: return [[OUT]], [[SCALE]], [[ZP]] : tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>
}

// -----

// CHECK-LABEL: @FuseDQSigned
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x304x560xf32>
func.func @FuseDQSigned(%arg0: tensor<1x304x560xf32>) -> (tensor<1x304x560xsi8>, tensor<1xf32>, tensor<1xsi8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<254.> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<0.00393700786> : tensor<f32>, [#const.Reshape<[1]>]
    %cst_1 = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32>, [#const.Reshape<[1]>]
    %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2]} : tensor<1x304x560xf32> -> tensor<1xf32>
    %1 = IE.Clamp(%0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Subtract(%cst_1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %3 = IE.ReduceMax(%arg0) {axes_value = [0, 1, 2]} : tensor<1x304x560xf32> -> tensor<1xf32>
    %4 = IE.Clamp(%3) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1xf32> -> tensor<1xf32>
    %5 = IE.Subtract(%4, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %6 = IE.Multiply(%5, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %7 = IE.Divide(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %8 = IE.Round(%7) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1xf32> -> tensor<1xf32>
    %9 = IE.Clamp(%8) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1xf32> -> tensor<1xf32>
    %10 = IE.Convert(%9) {dstElemType = si8} : tensor<1xf32> -> tensor<1xsi8>
    %11 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1x1x1xf32> -> tensor<1x304x560xf32>
    %12 = IE.Divide(%11, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1xf32> -> tensor<1x304x560xf32>
    %13 = IE.Round(%12) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x304x560xf32> -> tensor<1x304x560xf32>
    %14 = IE.Add(%13, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1xf32> -> tensor<1x304x560xf32>
    %15 = IE.Clamp(%14) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1x304x560xf32> -> tensor<1x304x560xf32>
    %16 = IE.Convert(%15) {dstElemType = si8} : tensor<1x304x560xf32> -> tensor<1x304x560xsi8>
    return %16, %6, %10 : tensor<1x304x560xsi8>, tensor<1xf32>, tensor<1xsi8>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add
    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]]) {dstElemType = si8
    // CHECK-SAME: -> tensor<1x304x560xsi8>, tensor<1xf32>, tensor<1xsi8>
    // CHECK: return [[OUT]], [[SCALE]], [[ZP]] : tensor<1x304x560xsi8>, tensor<1xf32>, tensor<1xsi8>
}

// -----

// CHECK-LABEL: @FuseDQSpanUnsigned
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x4x8xf32>
func.func @FuseDQSpanUnsigned(%arg0: tensor<1x4x8xf32>) -> (tensor<1x4x8xui8>, tensor<1xf32>, tensor<1xui8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<255.> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<255.> : tensor<f32>, [#const.Reshape<[1]>]
    %cst_1 = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32>, [#const.Reshape<[1]>]
    %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2]} : tensor<1x4x8xf32> -> tensor<1xf32>
    %1 = IE.Clamp(%0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Subtract(%cst_1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %3 = IE.ReduceMax(%arg0) {axes_value = [0, 1, 2]} : tensor<1x4x8xf32> -> tensor<1xf32>
    %4 = IE.Clamp(%3) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1xf32> -> tensor<1xf32>
    %5 = IE.Subtract(%4, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %6 = IE.Divide(%5, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %7 = IE.Divide(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %8 = IE.Round(%7) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1xf32> -> tensor<1xf32>
    %9 = IE.Clamp(%8) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1xf32> -> tensor<1xf32>
    %10 = IE.Convert(%9) {dstElemType = ui8} : tensor<1xf32> -> tensor<1xui8>
    %11 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1x1x1xf32> -> tensor<1x4x8xf32>
    %12 = IE.Divide(%11, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1xf32> -> tensor<1x4x8xf32>
    %13 = IE.Round(%12) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x4x8xf32> -> tensor<1x4x8xf32>
    %14 = IE.Add(%13, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1xf32> -> tensor<1x4x8xf32>
    %15 = IE.Clamp(%14) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x4x8xf32> -> tensor<1x4x8xf32>
    %16 = IE.Convert(%15) {dstElemType = ui8} : tensor<1x4x8xf32> -> tensor<1x4x8xui8>
    return %16, %6, %10 : tensor<1x4x8xui8>, tensor<1xf32>, tensor<1xui8>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add
    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]])
    // CHECK-SAME: -> tensor<1x4x8xui8>, tensor<1xf32>, tensor<1xui8>
    // CHECK: return [[OUT]], [[SCALE]], [[ZP]] : tensor<1x4x8xui8>, tensor<1xf32>, tensor<1xui8>
}

// -----

// CHECK-LABEL: @FuseDQSpanSigned
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x4x8xf32>
func.func @FuseDQSpanSigned(%arg0: tensor<1x4x8xf32>) -> (tensor<1x4x8xsi8>, tensor<1xf32>, tensor<1xsi8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<254.> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<254.> : tensor<f32>, [#const.Reshape<[1]>]
    %cst_1 = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32>, [#const.Reshape<[1]>]
    %0 = IE.ReduceMin(%arg0) {axes_value = [0, 1, 2]} : tensor<1x4x8xf32> -> tensor<1xf32>
    %1 = IE.Clamp(%0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1xf32> -> tensor<1xf32>
    %2 = IE.Subtract(%cst_1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %3 = IE.ReduceMax(%arg0) {axes_value = [0, 1, 2]} : tensor<1x4x8xf32> -> tensor<1xf32>
    %4 = IE.Clamp(%3) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1xf32> -> tensor<1xf32>
    %5 = IE.Subtract(%4, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %6 = IE.Divide(%5, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %7 = IE.Divide(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %8 = IE.Round(%7) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1xf32> -> tensor<1xf32>
    %9 = IE.Clamp(%8) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1xf32> -> tensor<1xf32>
    %10 = IE.Convert(%9) {dstElemType = si8} : tensor<1xf32> -> tensor<1xsi8>
    %11 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1x1x1xf32> -> tensor<1x4x8xf32>
    %12 = IE.Divide(%11, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1xf32> -> tensor<1x4x8xf32>
    %13 = IE.Round(%12) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x4x8xf32> -> tensor<1x4x8xf32>
    %14 = IE.Add(%13, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1xf32> -> tensor<1x4x8xf32>
    %15 = IE.Clamp(%14) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1x4x8xf32> -> tensor<1x4x8xf32>
    %16 = IE.Convert(%15) {dstElemType = si8} : tensor<1x4x8xf32> -> tensor<1x4x8xsi8>
    return %16, %6, %10 : tensor<1x4x8xsi8>, tensor<1xf32>, tensor<1xsi8>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add
    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]]) {dstElemType = si8
    // CHECK-SAME: -> tensor<1x4x8xsi8>, tensor<1xf32>, tensor<1xsi8>
    // CHECK: return [[OUT]], [[SCALE]], [[ZP]] : tensor<1x4x8xsi8>, tensor<1xf32>, tensor<1xsi8>
}

// -----

// CHECK-LABEL: @FuseDQPerAxis
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x4x8xf32>
func.func @FuseDQPerAxis(%arg0: tensor<1x4x8xf32>) -> (tensor<1x4x8xui8>, tensor<1x4x1xf32>, tensor<1x4x1xui8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<255.> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1xf32> = dense<0.0039215> : tensor<1x1x1xf32>
    %cst_1 = const.Declare tensor<1x1x1xf32> = dense<0.0> : tensor<1x1x1xf32>
    %0 = IE.ReduceMin(%arg0) {axes_value = [2], keep_dims} : tensor<1x4x8xf32> -> tensor<1x4x1xf32>
    %1 = IE.Clamp(%0) {max = 0.000000e+00 : f64, min = -6.550400e+04 : f64} : tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %2 = IE.Subtract(%cst_1, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf32>, tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %3 = IE.ReduceMax(%arg0) {axes_value = [2], keep_dims} : tensor<1x4x8xf32> -> tensor<1x4x1xf32>
    %4 = IE.Clamp(%3) {max = 6.550400e+04 : f64, min = 0.000000e+00 : f64} : tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %5 = IE.Subtract(%4, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1xf32>, tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %6 = IE.Multiply(%5, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1xf32>, tensor<1x1x1xf32> -> tensor<1x4x1xf32>
    %7 = IE.Divide(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x1xf32>, tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %8 = IE.Round(%7) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %9 = IE.Clamp(%8) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x4x1xf32> -> tensor<1x4x1xf32>
    %10 = IE.Convert(%9) {dstElemType = ui8} : tensor<1x4x1xf32> -> tensor<1x4x1xui8>
    %11 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1x1x1xf32> -> tensor<1x4x8xf32>
    %12 = IE.Divide(%11, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1x4x1xf32> -> tensor<1x4x8xf32>
    %13 = IE.Round(%12) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x4x8xf32> -> tensor<1x4x8xf32>
    %14 = IE.Add(%13, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x8xf32>, tensor<1x4x1xf32> -> tensor<1x4x8xf32>
    %15 = IE.Clamp(%14) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x4x8xf32> -> tensor<1x4x8xf32>
    %16 = IE.Convert(%15) {dstElemType = ui8} : tensor<1x4x8xf32> -> tensor<1x4x8xui8>
    return %16, %6, %10 : tensor<1x4x8xui8>, tensor<1x4x1xf32>, tensor<1x4x1xui8>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add
    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]])
    // CHECK-SAME: -> tensor<1x4x8xui8>, tensor<1x4x1xf32>, tensor<1x4x1xui8>
    // CHECK: return [[OUT]], [[SCALE]], [[ZP]] : tensor<1x4x8xui8>, tensor<1x4x1xf32>, tensor<1x4x1xui8>
}

// -----

// CHECK-LABEL: @FuseDQSignedSymmetricClamp
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x2x1x64xf32>
func.func @FuseDQSignedSymmetricClamp(%arg0: tensor<1x2x1x64xf32>) -> (tensor<1x2x1x64xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>) {
    %cst = const.Declare tensor<1x1x1x1xf32> = dense<3.93700786e-03> : tensor<1x1x1x1xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
    %0 = IE.ReduceMin(%arg0) {axes_value = [3], keep_dims} : tensor<1x2x1x64xf32> -> tensor<1x2x1x1xf32>
    %1 = IE.Clamp(%0) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf32>
    %2 = IE.Multiply(%1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x1x1xf32>
    %3 = IE.ReduceMax(%arg0) {axes_value = [3], keep_dims} : tensor<1x2x1x64xf32> -> tensor<1x2x1x1xf32>
    %4 = IE.Clamp(%3) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf32>
    %5 = IE.Subtract(%4, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf32>
    %6 = IE.Multiply(%5, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x2x1x1xf32>
    %7 = IE.Divide(%2, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf32>
    %8 = IE.Round(%7) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf32>
    %9 = IE.Clamp(%8) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1x2x1x1xf32> -> tensor<1x2x1x1xf32>
    %10 = IE.Divide(%arg0, %6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x64xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x1x64xf32>
    %11 = IE.Round(%10) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x2x1x64xf32> -> tensor<1x2x1x64xf32>
    %12 = IE.Add(%11, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x64xf32>, tensor<1x2x1x1xf32> -> tensor<1x2x1x64xf32>
    %13 = IE.Clamp(%12) {max = 1.270000e+02 : f64, min = -1.270000e+02 : f64} : tensor<1x2x1x64xf32> -> tensor<1x2x1x64xf32>
    return %13, %6, %9 : tensor<1x2x1x64xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>

    // CHECK-NOT: IE.Multiply
    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add
    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT_I8:%.+]], [[SCALE:%.+]], [[ZP_I8:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]]) {dstElemType = si8
    // CHECK-SAME: -> tensor<1x2x1x64xsi8>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xsi8>
    // CHECK: [[ZP_F32:%.+]] = IE.Convert([[ZP_I8]]) {dstElemType = f32}
    // CHECK: [[OUT_F32:%.+]] = IE.Convert([[OUT_I8]]) {dstElemType = f32}
    // CHECK: return [[OUT_F32]], [[SCALE]], [[ZP_F32]] : tensor<1x2x1x64xf32>, tensor<1x2x1x1xf32>, tensor<1x2x1x1xf32>
}

// -----

// The scale (quant_scales_mul) feeds a downstream consumer that is placed in the block before the
// activation-quant output Convert -- the same topology as the SAM decoder cross-attention MatMuls.
// The fused IE.DynamicQuantize and its result Converts must be inserted early enough (after the
// reduce/clamp ops that produce its min/max inputs) so that the new scale value dominates every
// replaced use. Inserting at the tail of the decomposition chain would place the scale after this
// earlier consumer and break SSA dominance ('operand does not dominate this use').

// CHECK-LABEL: @FuseDQScaleUsedBeforeOutputConvert
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x304x560xf32>
func.func @FuseDQScaleUsedBeforeOutputConvert(%arg0: tensor<1x304x560xf32>) -> (tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>) {
    %cst = const.Declare tensor<1x1x1xf32> = dense<255.> : tensor<1x1x1xf32>
    %cst_0 = const.Declare tensor<1xf32> = dense<0.0039215> : tensor<f32>, [#const.Reshape<[1]>]
    %cst_1 = const.Declare tensor<1xf32> = dense<0.0> : tensor<f32>, [#const.Reshape<[1]>]
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
    // External consumer of the scale, positioned before the activation output Convert (%16).
    %scaleUser = IE.Multiply(%6, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
    %11 = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1x1x1xf32> -> tensor<1x304x560xf32>
    %12 = IE.Divide(%11, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1xf32> -> tensor<1x304x560xf32>
    %13 = IE.Round(%12) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x304x560xf32> -> tensor<1x304x560xf32>
    %14 = IE.Add(%13, %9) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x304x560xf32>, tensor<1xf32> -> tensor<1x304x560xf32>
    %15 = IE.Clamp(%14) {max = 2.550000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x304x560xf32> -> tensor<1x304x560xf32>
    %16 = IE.Convert(%15) {dstElemType = ui8} : tensor<1x304x560xf32> -> tensor<1x304x560xui8>
    return %16, %scaleUser, %10 : tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>

    // CHECK-NOT: IE.Subtract
    // CHECK-NOT: IE.Divide
    // CHECK-NOT: IE.Add
    // CHECK-NOT: IE.Round

    // CHECK: [[ReduceMin:%.+]] = IE.ReduceMin([[INPUT]])
    // CHECK: [[ClampMin:%.+]] = IE.Clamp([[ReduceMin]])
    // CHECK: [[ReduceMax:%.+]] = IE.ReduceMax([[INPUT]])
    // CHECK: [[ClampMax:%.+]] = IE.Clamp([[ReduceMax]])
    // CHECK: [[OUT:%.+]], [[SCALE:%.+]], [[ZP:%.+]] = IE.DynamicQuantize([[INPUT]], [[ClampMin]], [[ClampMax]])
    // CHECK-SAME: -> tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>
    // The surviving scale consumer must reference the fused DynamicQuantize scale result.
    // CHECK: [[SCALE_USER:%.+]] = IE.Multiply([[SCALE]], %cst{{.*}})
    // CHECK: return [[OUT]], [[SCALE_USER]], [[ZP]] : tensor<1x304x560xui8>, tensor<1xf32>, tensor<1xui8>
}
