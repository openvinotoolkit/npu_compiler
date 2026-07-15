//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --platform=%platform% --setup-npu-constraint %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @mainModule attributes { config.platform = #config.platform<NPU4000> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @config.UseDedicatedFifoPerShaveEngine : false
// CHECK: config.Option @config.BarrierMaxSlotCount : 256
// CHECK: config.Option @config.MetadataMaxKernelInvocationCount : 64
// CHECK: config.Option @config.MetadataMaxKernelRangeCount : 64
