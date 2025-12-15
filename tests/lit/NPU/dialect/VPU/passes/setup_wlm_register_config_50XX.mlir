//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --vpu-arch=%arch% --setup-npu-constraint %s | FileCheck %s
// REQUIRES: arch-NPU50XX

module @mainModule attributes { config.arch = #config.arch_kind<NPU50XX> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @config.DpuFIFOAddrs : [788529152, 788529184, 788529216, 788529248]
// CHECK: config.Option @config.ShvFIFOAddrs : [788578304, 788578336, 788578368, 788578400, 788578432, 788578464, 788578496, 788578528]
// CHECK: config.Option @config.BarrierFIFOAddr : 788594688 : ui64
