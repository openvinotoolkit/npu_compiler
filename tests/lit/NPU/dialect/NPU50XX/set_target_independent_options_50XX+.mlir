//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --set-target-independent-options="enable-sprlut" %s | FileCheck --check-prefix=CHECK-SPRLUT %s --strict-whitespace
// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --set-target-independent-options="weights-table-reuse-mode=ENABLED" %s | FileCheck --check-prefix=CHECK-WEIGHTSTABLE %s --strict-whitespace
// REQUIRES: arch-NPU50XX

module @mainModule attributes {} {
}

// CHECK-SPRLUT: module @mainModule
// CHECK-SPRLUT: config.PipelineOptions @Options
// CHECK-SPRLUT: config.Option @config.SprLUTEnabled : true
// CHECK-WEIGHTSTABLE: config.Option @config.WeightsTableReuseMode : 0

// -----

module @mainModule attributes {} {
}

// CHECK-WEIGHTSTABLE: module @mainModule
// CHECK-WEIGHTSTABLE: config.PipelineOptions @Options
// CHECK-WEIGHTSTABLE: config.Option @config.WeightsTableReuseMode : 0
// CHECK-SPRLUT: config.Option @config.SprLUTEnabled : true
