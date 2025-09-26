//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --vpu-arch=%arch% --setup-npu-constraint %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule attributes { config.arch = #config.arch_kind<NPU40XX> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @VPU.UseDedicatedFifoPerShaveEngine : false
// CHECK: config.Option @VPU.BarrierMaxVariantSum : 64
// CHECK: config.Option @VPU.BarrierMaxVariantCount : 128
// CHECK: config.Option @VPU.MetadataMaxKernelInvocationCount : 64
// CHECK: config.Option @VPU.MetadataMaxKernelRangeCount : 64
