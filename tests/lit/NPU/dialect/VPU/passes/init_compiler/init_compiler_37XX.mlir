//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// RUN: vpux-opt --init-compiler="platform=%platform%" %s | FileCheck %s
// REQUIRES: platform-NPU3720

module @mainModule {
}

// CHECK: module @mainModule attributes
// CHECK-NEXT: config.PipelineOptions @Options
// CHECK: config.Option @config.EnableProfiling : false
