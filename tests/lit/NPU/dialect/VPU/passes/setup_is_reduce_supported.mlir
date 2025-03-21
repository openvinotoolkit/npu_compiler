//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --setup-is-reduce-supported="enable-is-reduce-supported" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @mainModule attributes {} {
}

// CHECK: module @mainModule
// CHECK: IE.PipelineOptions @Options
// CHECK: IE.Option @VPU.ReduceSupported : true
