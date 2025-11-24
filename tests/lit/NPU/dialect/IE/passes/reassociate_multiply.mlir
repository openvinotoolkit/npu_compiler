//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --reassociate-multiply %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: func.func @ReassociateMultiply
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x32x1x1xf16>, [[INPUT1:%.+]]: tensor<1x32x1024x96xf16>)
func.func @ReassociateMultiply(%arg0: tensor<1x32x1x1xf16>, %arg1: tensor<1x32x1024x96xf16>) -> tensor<1x32x1024x96xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>

  return %1 : tensor<1x32x1024x96xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1x1xf16>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[INPUT1]], [[MULTIPLY_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1024x96xf16>
  // CHECK: return [[MULTIPLY_2]]
}

// -----

// CHECK-LABEL: func.func @NotReassociateMultiplySameInputSize
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x32x1024x96xf16>, [[INPUT1:%.+]]: tensor<1x32x1024x96xf16>)
func.func @NotReassociateMultiplySameInputSize(%arg0: tensor<1x32x1024x96xf16>, %arg1: tensor<1x32x1024x96xf16>) -> tensor<1x32x1024x96xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>

  return %1 : tensor<1x32x1024x96xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
  //CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  //CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>
  //CHECK: return [[MULTIPLY_2]]
}

// -----

// CHECK-LABEL: func.func @NotReassociateMultiplyWithPostOp
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x32x1x1xf16>, [[INPUT1:%.+]]: tensor<1x32x1024x96xf16>)
func.func @NotReassociateMultiplyWithPostOp(%arg0: tensor<1x32x1x1xf16>, %arg1: tensor<1x32x1024x96xf16>) -> tensor<1x32x1024x96xf16> {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x32x1x1xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>

  return %1 : tensor<1x32x1024x96xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x32x1x1xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>
  // CHECK: return [[MULTIPLY_2]]
}

// -----

// CHECK-LABEL: func.func @NotReassociateMultiplyCanNotBroadcast
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x32x1x1xf16>, [[INPUT1:%.+]]: tensor<1x32x1024x96xf16>)
func.func @NotReassociateMultiplyCanNotBroadcast(%arg0: tensor<1x32x1x1xf16>, %arg1: tensor<1x32x1024x96xf16>) -> tensor<1x32x1024x96xf16> {
  %cst = const.Declare tensor<1x1x1x96xf16> = dense<2.0> : tensor<1x1x1x96xf16>

  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x32x1x1xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x96xf16> -> tensor<1x32x1024x96xf16>

  return %1 : tensor<1x32x1024x96xf16>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x96xf16> = dense<2.000000e+00> : tensor<1x1x1x96xf16>
  // CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<>} : tensor<1x32x1x1xf16>, tensor<1x32x1024x96xf16> -> tensor<1x32x1024x96xf16>
  // CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x96xf16> -> tensor<1x32x1024x96xf16>
  // CHECK: return [[MULTIPLY_2]]
}

// -----

// CHECK-LABEL: func.func @NotReassociateMultiplyMultiUses
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x32x1024x96xf16>, [[INPUT1:%.+]]: tensor<1x1x1x1xf16>)
func.func @NotReassociateMultiplyMultiUses(%arg0: tensor<1x32x1024x96xf16>, %arg1: tensor<1x1x1x1xf16>) -> (tensor<1x32x1024x96xf16>, tensor<1x32x1024x96xf16>) {
  %cst = const.Declare tensor<1x1x1x1xf16> = dense<2.0> : tensor<1x1x1x1xf16>

  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>
  %1 = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>

  return %0, %1 : tensor<1x32x1024x96xf16>, tensor<1x32x1024x96xf16>

  //CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1x1xf16>
  //CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>
  //CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x32x1024x96xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x1024x96xf16>
  //CHECK: return [[MULTIPLY_1]], [[MULTIPLY_2]]
}

// -----

// CHECK-LABEL: func.func @NotReassociateMultiplyForTooBigOutput
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x1536x16xf16>, [[INPUT1:%.+]]: tensor<1x1x16xf16>, [[INPUT2:%.+]]: tensor<1x1536x1xf16>)
func.func @NotReassociateMultiplyForTooBigOutput(%arg0: tensor<1x1536x16xf16>, %arg1: tensor<1x1x16xf16>, %arg2: tensor<1x1536x1xf16>) -> tensor<1x1536x16xf16> {
  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x16xf16>, tensor<1x1x16xf16> -> tensor<1x1536x16xf16>
  %1 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x16xf16>, tensor<1x1536x1xf16> -> tensor<1x1536x16xf16>

  return %1 : tensor<1x1536x16xf16>

  //CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x16xf16>, tensor<1x1x16xf16> -> tensor<1x1536x16xf16>
  //CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x16xf16>, tensor<1x1536x1xf16> -> tensor<1x1536x16xf16>

  //CHECK: return [[MULTIPLY_2]]
}

// -----

// CHECK-LABEL: func.func @NotReassociateMultiplyForTooBigOutputInputBroadcasted
// CHECK-SAME:  ([[INPUT0:%.+]]: tensor<1x1536x1xf16>, [[INPUT1:%.+]]: tensor<1x1x16xf16>, [[INPUT2:%.+]]: tensor<1x1536x1xf16>)
func.func @NotReassociateMultiplyForTooBigOutputInputBroadcasted(%arg0: tensor<1x1536x1xf16>, %arg1: tensor<1x1x16xf16>, %arg2: tensor<1x1536x1xf16>) -> tensor<1x1536x16xf16> {
  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x1xf16>, tensor<1x1x16xf16> -> tensor<1x1536x16xf16>
  %1 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x16xf16>, tensor<1x1536x1xf16> -> tensor<1x1536x16xf16>

  return %1 : tensor<1x1536x16xf16>

  //CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x1xf16>, tensor<1x1x16xf16> -> tensor<1x1536x16xf16>
  //CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[MULTIPLY_1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1536x16xf16>, tensor<1x1536x1xf16> -> tensor<1x1536x16xf16>

  //CHECK: return [[MULTIPLY_2]]
}

// -----
// CHECK-LABEL: func.func @ReassociateMultiplyDynamicShape
// CHECK-SAME: ([[INPUT0:%.+]]: tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>, [[INPUT1:%.+]]: tensor<1x16x1x1xf16>, [[INPUT2:%.+]]: tensor<1x16x1x1xf16>)
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @ReassociateMultiplyDynamicShape(%arg0: tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>,  %arg1: tensor<1x16x1x1xf16>, %arg2: tensor<1x16x1x1xf16>) -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}> {
  %0 = IE.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x1x1xf16> -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>
  %1 = IE.Multiply(%0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x1x1xf16> -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>
  return %1: tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>

  //CHECK: [[MULTIPLY_1:%.+]] = IE.Multiply([[INPUT1]], [[INPUT2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x1x1xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x1x1xf16>
  //CHECK: [[MULTIPLY_2:%.+]] = IE.Multiply([[INPUT0]], [[MULTIPLY_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>, tensor<1x16x1x1xf16> -> tensor<1x16x?x1920xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 1080, 1920]> : tensor<4xsi64>, order = #NCHW}>

  //CHECK:  return [[MULTIPLY_2]]
}
