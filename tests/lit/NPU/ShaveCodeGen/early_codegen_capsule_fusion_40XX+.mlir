//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --encapsulate-codegen-ops --early-codegen-capsule-fusion %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @SingleCosNoChange {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res : tensor<1x1x1x1000xf16>

    // Single node -> does not alter the capsule
    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosFuse {
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

    // Eltwise chain -> full fusion is expected
    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES1:%.+]] = IE.Cos([[COS_RES]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES2:%.+]] = IE.Cos([[COS_RES1]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES2]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosWithChainBreak {
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
    %cos_res3 = IE.Cos(%cos_res2) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    return %cos_res3 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES1:%.+]] = IE.Cos([[COS_RES]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES1]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // Unsupported layer should stay unwrapped & fusion chain should be interrupted
    // CHECK: [[SIG_RES:%.+]] = IE.Sigmoid([[VAR0]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[SIG_RES]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES1:%.+]] = IE.Cos([[COS_RES]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES1]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR1]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @MultipleCosWithFork {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // Expected chain 0
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // Expected chain 1
    %cos_res2 = IE.Cos(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res3 = IE.Cos(%cos_res2) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // Expected chain 2
    %cos_res4 = IE.Cos(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res5 = IE.Cos(%cos_res4) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    return %cos_res5 : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES1:%.+]] = IE.Cos([[COS_RES]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES1]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES1:%.+]] = IE.Cos([[COS_RES]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES1]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[COS_RES1:%.+]] = IE.Cos([[COS_RES]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES1]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @BinaryFusableEltwiseLayers {
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
    // CHECK: [[DIV_RES:%.+]] = IE.Divide([[MAX_RES]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[DIV_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @FusableEltwiseLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "max" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %div_res = IE.Divide(%arg1, %cos_res) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %div2_res = IE.Divide(%div_res, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    return %div2_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG0]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: [[DIV0_RES:%.+]] = IE.Divide([[CAPSULE_ARG1]], [[COS_RES]])
    // CHECK: [[DIV1_RES:%.+]] = IE.Divide([[DIV0_RES]], [[CAPSULE_ARG0]])
    // CHECK: IE.CGCYield [[DIV1_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @BinaryNonFusableEltwiseLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "max" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %max_res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    // Divide op has dependencies on both the results of Maximum & Cos, thus any of the Max+Div or Cos+Div fusions might introduce stalls
    // Ideally, a 3-way fusion should take place (Max+Cos->Divide), but current infra only analyzes 2-part sequential fusions, so no fusion is expected for this case
    %div_res = IE.Divide(%max_res, %cos_res) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %div_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[MAX_RES:%.+]] = IE.Maximum([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[MAX_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR1:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[CAPSULE_ARG]]) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: [[VAR2:%.+]] = IE.CodeGenCapsule inputs([[VAR0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[VAR1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[DIV_RES:%.+]] = IE.Divide([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: IE.CGCYield [[DIV_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>

    // CHECK: return [[VAR2]] : tensor<1x1x1x1000xf16>
  }
}

// -----

module @BinaryPlusUnaryFusableEltwiseLayers {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
    DataInfo "input1" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "max" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>, %arg1: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    %max_res = IE.Maximum(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT> } : tensor<1x1x1x1000xf16>, tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res = IE.Cos(%max_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>

    return %cos_res : tensor<1x1x1x1000xf16>

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1:%.+]]: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> {
    // CHECK: [[VAR0:%.+]] = IE.CodeGenCapsule inputs([[ARG0]] as [[CAPSULE_ARG0:%.+]]: tensor<1x1x1x1000xf16>, [[ARG1]] as [[CAPSULE_ARG1:%.+]]: tensor<1x1x1x1000xf16>) {
    // CHECK: [[MAX_RES:%.+]] = IE.Maximum([[CAPSULE_ARG0]], [[CAPSULE_ARG1]])
    // CHECK: [[COS_RES:%.+]] = IE.Cos([[MAX_RES]])
    // CHECK: IE.CGCYield [[COS_RES]] : tensor<1x1x1x1000xf16>
    // CHECK: } -> tensor<1x1x1x1000xf16>
    // CHECK: return [[VAR0]] : tensor<1x1x1x1000xf16>
  }
}
