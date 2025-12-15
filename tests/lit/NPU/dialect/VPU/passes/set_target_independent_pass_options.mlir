//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --set-target-independent-options="enable-auto-padding-odu enable-auto-padding-idu" %s | FileCheck --check-prefix=CHECK-PAD %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX
// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --set-target-independent-options="enable-is-reduce-supported " %s | FileCheck --check-prefix=CHECK-REDUCE %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX
// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --set-target-independent-options="allow-custom-values=true" %s | FileCheck --check-prefix=CHECK-CUSTOM %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

module @mainModule attributes {} {
}

// CHECK-PAD: module @mainModule
// CHECK-PAD: config.PipelineOptions @Options
// CHECK-PAD: config.Option @config.AutoPaddingODU : true
// CHECK-PAD: config.Option @config.AutoPaddingIDU : true
// CHECK-REDUCE: config.Option @config.ReduceSupported : true

// -----

module @mainModule attributes {} {
}


// CHECK-REDUCE: module @mainModule
// CHECK-REDUCE: config.PipelineOptions @Options
// CHECK-REDUCE: config.Option @config.ReduceSupported : true
// CHECK-PAD: config.Option @config.AutoPaddingIDU : true

// -----

module @NoInsertionNeeded {
  config.PipelineOptions @Options {
    config.Option @VPU.MyOptions: false
  }
}


// CHECK-CUSTOM: module @NoInsertionNeeded
// CHECK-CUSTOM: config.PipelineOptions @Options
// CHECK-REDUCE: config.Option @config.ReduceSupported : true
// CHECK-PAD: config.Option @config.AutoPaddingIDU : true
// CHECK-CUSTOM: config.Option @VPU.MyOptions : false
