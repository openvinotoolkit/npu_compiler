//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

/// InitCompilerOptions has enable-adaptive-stripping as a flag and adaptive stripping can also be set using
/// NPU_QDQ_OPTIMIZATION.
/// This test checks to make sure that default-hw-mode for non specific dialects cannot invoke
/// enable-adaptive-stripping flag as it was removed to remove ambiguity for which flag takes priority.
/// stderr should be: <Pass-Options-Parser>: no such option enable-adaptive-stripping

// XFAIL: *
// RUN: vpux-opt --vpu-arch=%arch% --default-hw-mode="enable-adaptive-stripping" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

/// This is a dummy module with a dummy function to run the LIT test and expect a fail with
/// enable-adaptive-stripping passed through default-hw-mode
module {
  net.NetworkInfo entryPoint : @main
  inputsInfo : {
    DataInfo "input" : tensor<1x1x1x1xf32>
  } outputsInfo : {
    DataInfo "output" : tensor<1x1x1x1xf32>
  }

  func.func @main(%arg0: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
    return %arg0 : tensor<1x1x1x1xf32>
  }
}

/// Dummy CHECK exists for the purpose of needing at least one of them in this file
// CHECK-DAG: #config.compilation_mode<DefaultHW>
