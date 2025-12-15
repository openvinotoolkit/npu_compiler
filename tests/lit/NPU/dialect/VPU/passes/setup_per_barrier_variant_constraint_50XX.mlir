//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --vpu-arch=%arch% --setup-npu-constraint="workload-management-status=ENABLED enable-sw-kernel-fifo-per-shave-engine=true" %s | FileCheck %s
// REQUIRES: arch-NPU50XX

module @mainModule attributes { config.arch = #config.arch_kind<NPU50XX> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK-DAG: config.Option @config.UseDedicatedFifoPerShaveEngine : true
// CHECK-DAG: config.Option @config.BarrierMaxVariantSum : 64
// CHECK-DAG: config.Option @config.BarrierMaxVariantCount : 128
// CHECK-DAG: config.Option @config.MetadataMaxKernelInvocationCount : 32
// CHECK-DAG: config.Option @config.MetadataMaxKernelRangeCount : 32
// CHECK-DAG: config.Option @config.WorkloadManagementStatus : "ENABLED"
