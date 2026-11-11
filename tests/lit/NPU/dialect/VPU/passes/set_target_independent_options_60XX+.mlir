//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --set-target-independent-options="enable-dcim power-optimized" %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU6010

module @mainModule attributes { } {
}

// CHECK: module @mainModule
// CHECK: config.PipelineOptions @Options
// CHECK: config.Option @config.PowerOptimized : true
// CHECK: config.Option @config.EnableDCIM : true
