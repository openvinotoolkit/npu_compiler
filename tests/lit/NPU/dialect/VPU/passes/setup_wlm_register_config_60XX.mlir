//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --setup-npu-constraint %s | FileCheck %s
// REQUIRES: platform-NPU6010

module @mainModule attributes { config.platform = #config.platform<NPU6010> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @config.DpuFIFOAddrs : [788529152, 788529184, 788529216, 788529248]
// CHECK: config.Option @config.ShvFIFOAddrs : [788578304, 788578336, 788578432, 788578464, 788578560, 788578592, 788578688, 788578720]
// CHECK: config.Option @config.BarrierFIFOAddr : 788594688 : ui64
