//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --platform=%platform% --setup-npu-constraint="enable-sw-kernel-fifo-per-shave-engine=true" %s | FileCheck %s
// REQUIRES: platform-NPU5010

module @mainModule attributes { config.platform = #config.platform<NPU5010> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK-DAG: config.Option @config.UseDedicatedFifoPerShaveEngine : true
// CHECK-DAG: config.Option @config.BarrierMaxSlotCount : 256
// CHECK-DAG: config.Option @config.MetadataMaxKernelInvocationCount : 32
// CHECK-DAG: config.Option @config.MetadataMaxKernelRangeCount : 32
