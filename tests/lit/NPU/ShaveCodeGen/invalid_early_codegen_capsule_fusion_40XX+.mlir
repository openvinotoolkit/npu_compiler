//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// The EarlyCodegenCapsuleFusion pass relies on FusionChainAnalysis to be pre-computed & cached
// So if there is no analysis cached (as in no previous pass perfomed it), the pass is expected to fail

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --early-codegen-capsule-fusion %s -verify-diagnostics
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @InvalidMultipleCosFuse {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo : {
    DataInfo "cos" : tensor<1x1x1x1000xf16>
  }

  func.func @main(%arg0: tensor<1x1x1x1000xf16>) -> tensor<1x1x1x1000xf16> { // expected-error {{FusionChainAnalysis is expected to be cached}}
    %cos_res = IE.Cos(%arg0) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res1 = IE.Cos(%cos_res) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    %cos_res2 = IE.Cos(%cos_res1) : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
    return %cos_res2 : tensor<1x1x1x1000xf16>
  }
}
