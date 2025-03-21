//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --vpu-arch=%arch% --setup-npu-constraint %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @mainModule attributes { VPU.arch = #VPU.arch_kind<NPU40XX> } {
}

// CHECK: module @mainModule attributes
// CHECK: IE.PipelineOptions @Options
// CHECK: IE.Option @VPU.BarrierMaxVariantSum : 64
// CHECK: IE.Option @VPU.BarrierMaxVariantCount : 128
