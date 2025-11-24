//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --setup-npu-constraint="workload-management-status=ENABLED enable-sw-kernel-fifo-per-shave-engine=true" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule attributes { config.arch = #config.arch_kind<NPU40XX> } {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK-DAG: config.Option @config.UseDedicatedFifoPerShaveEngine : true
// Currently, still non-WLM barrier configuration settings are present even for WLM enabled mode. To be updated to WLM values in E#155846
// CHECK-DAG: config.Option @config.BarrierMaxVariantSum : 64
// CHECK-DAG: config.Option @config.BarrierMaxVariantCount : 128
// CHECK-DAG: config.Option @config.MetadataMaxVariantCount : 128
// CHECK-DAG: config.Option @config.MetadataMaxInvariantCount : 64
// CHECK-DAG: config.Option @config.MetadataMaxKernelInvocationCount : 32
// CHECK-DAG: config.Option @config.MetadataMaxKernelRangeCount : 32
// CHECK-DAG: config.Option @config.WorkloadManagementStatus : "ENABLED"
