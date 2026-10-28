//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" \
// RUN:     --setup-task-descriptor-constraints="dpu-invariant-size=352 dpu-variant-size=224 act-kernel-invocation-size=96 act-kernel-range-size=40 dma-size=224 m2i-size=240" \
// RUN:     %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: module @WritesAllConstraints
// CHECK:   config.PipelineOptions @Options
// CHECK:     config.Option @config.TaskBinarySizeDPUInvariant : 352 : ui64
// CHECK:     config.Option @config.TaskBinarySizeDPUVariant : 224 : ui64
// CHECK:     config.Option @config.TaskBinarySizeActKernelInvocation : 96 : ui64
// CHECK:     config.Option @config.TaskBinarySizeActKernelRange : 40 : ui64
// CHECK:     config.Option @config.TaskBinarySizeDMA : 224 : ui64
// CHECK:     config.Option @config.TaskBinarySizeM2I : 240 : ui64
module @WritesAllConstraints {
}

// -----

// Verify that an option already present in PipelineOptions is not overwritten.
// CHECK-LABEL: module @PreservesExistingConstraint
// CHECK:   config.PipelineOptions @Options
// CHECK:     config.Option @config.TaskBinarySizeDPUVariant : 224 : ui64
// CHECK:     config.Option @config.TaskBinarySizeActKernelInvocation : 96 : ui64
// CHECK:     config.Option @config.TaskBinarySizeActKernelRange : 40 : ui64
// CHECK:     config.Option @config.TaskBinarySizeDMA : 224 : ui64
// CHECK:     config.Option @config.TaskBinarySizeM2I : 240 : ui64
// CHECK:     config.Option @config.TaskBinarySizeDPUInvariant : 99 : ui64
module @PreservesExistingConstraint {
  config.PipelineOptions @Options {
    config.Option @config.TaskBinarySizeDPUInvariant : 99 : ui64
  }
}
