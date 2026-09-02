//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --add-performance-metrics-section %s | FileCheck %s
// REQUIRES: platform-NPU4000

module @InsertPerformanceMetrics {
    func.func @main() {
        ELF.Main {
        }
        return
    }
}

// CHECK-LABEL: module @InsertPerformanceMetrics
// CHECK: func.func @main()
// CHECK: ELF.Main {
// CHECK: ELF.CreateSection @perf.metrics
// CHECK: ELF.PerformanceMetricsSection @PerfMetrics
