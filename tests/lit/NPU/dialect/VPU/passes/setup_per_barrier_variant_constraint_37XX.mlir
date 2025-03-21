//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --vpu-arch=%arch% --setup-npu-constraint %s | FileCheck %s
// REQUIRES: arch-NPU37XX

module @mainModule attributes { VPU.arch = #VPU.arch_kind<NPU37XX> } {
}

// CHECK: module @mainModule attributes
// CHECK: IE.PipelineOptions @Options
// CHECK: IE.Option @VPU.BarrierMaxVariantSum : 256
// CHECK: IE.Option @VPU.BarrierMaxVariantCount : 256
// CHECK: IE.Option @VPU.MetadataMaxVariantCount : 256
// CHECK: IE.Option @VPU.MetadataMaxInvariantCount : 32
// CHECK: IE.Option @VPU.MetadataMaxKernelInvocationCount : 64
// CHECK: IE.Option @VPU.MetadataMaxKernelRangeCount : 32
// CHECK-NOT: IE.Option @VPU.MetadataMaxMediaCount : 4
