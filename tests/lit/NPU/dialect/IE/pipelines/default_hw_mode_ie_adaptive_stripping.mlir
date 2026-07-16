//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/// Adaptive stripping is enabled by default.
/// Adaptive Stripping can be unset/disabled using NPU_QDQ_OPTIMIZATION from COMPILE_PARAMS
/// or through InitCompilerOptions using the flag enable-adaptive-stripping.

/// This test checks to make sure that default-hw-mode-ie cannot invoke enable-adaptive-stripping flag as it
/// was removed to remove ambiguity for which flag takes priority.
/// stderr should be: <Pass-Options-Parser>: no such option enable-adaptive-stripping

// XFAIL: *
// RUN: vpux-opt --platform=%platform% --default-hw-mode-ie="enable-adaptive-stripping" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

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
