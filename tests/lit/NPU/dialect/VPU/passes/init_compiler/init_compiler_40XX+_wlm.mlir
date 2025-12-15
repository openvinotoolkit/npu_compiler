//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU40XX || arch-NPU50XX

module @mainModule {
}

// CHECK: module @mainModule attributes
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @config.WorkloadManagementStatus : "ENABLED"
