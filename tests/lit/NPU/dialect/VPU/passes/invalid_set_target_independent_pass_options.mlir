//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --set-target-independent-options="allow-custom-values=false" --verify-diagnostics %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// expected-error@+1 {{Option config.SprLUTEnabled is already defined, probably you run '--init-compiler' twice}}
module @NoInsertionNeeded {
  config.PipelineOptions @Options {
    config.Option @config.SprLUTEnabled : false
    config.Option @VPU.MyOptions: false
  }
}
