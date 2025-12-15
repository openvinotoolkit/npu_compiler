//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --encapsulate-codegen-ops %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @SingleCosLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res2 = IE.Cos(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[VAR1]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosWithExceptionLayer {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    %sig = IE.Sigmoid(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16> // not supported by ShaveCodeGen, at least for now

    %cos_res2 = IE.Cos(%sig) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // Unsupported layer should stay unwrapped
    // CHECK: [[SIG_RES:%.+]] = IE.Sigmoid([[VAR1]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[SIG_RES]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @BinaryEltwiseLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "max" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %max_res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %div_res = IE.Divide(%max_res, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %div_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[MAX_RES:%.+]] = IE.Maximum([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[MAX_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[DIV_RES:%.+]] = IE.Divide([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[DIV_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR1]] : tensor<1x1x1x1000xf16>
  }
}
