//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --set-target-independent-options="allow-custom-values=false" --verify-diagnostics %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// expected-error@+1 {{Option config.SprLUTEnabled is already defined, probably you run '--init-compiler' twice}}
module @NoInsertionNeeded {
  config.PipelineOptions @Options {
    config.Option @config.SprLUTEnabled : false
    config.Option @VPU.MyOptions: false
  }
}
