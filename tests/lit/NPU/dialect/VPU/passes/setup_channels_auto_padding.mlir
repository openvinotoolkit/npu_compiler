//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --setup-channels-auto-padding="enable-auto-padding-odu enable-auto-padding-idu" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX
module @mainModule attributes {} {
}

// CHECK: module @mainModule
// CHECK: IE.PipelineOptions @Options
// CHECK: IE.Option @VPU.AutoPaddingODU : true
// CHECK: IE.Option @VPU.AutoPaddingIDU : true
