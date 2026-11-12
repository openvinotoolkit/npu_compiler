//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% enable-qdq-optimization-aggressive=false enable-adaptive-stripping=false" %s | FileCheck --check-prefix=CHECK-AGG-OFF-ADA-OFF %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-qdq-optimization-aggressive=false enable-adaptive-stripping=true" %s | FileCheck --check-prefix=CHECK-AGG-OFF-ADA-ON %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-qdq-optimization-aggressive=true enable-adaptive-stripping=false" %s | FileCheck --check-prefix=CHECK-AGG-ON-ADA-OFF %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-qdq-optimization-aggressive=true enable-adaptive-stripping=true" %s | FileCheck --check-prefix=CHECK-AGG-ON-ADA-ON %s
// RUN: vpux-opt --init-compiler="platform=%platform%" %s | FileCheck --check-prefix=CHECK-AGG-UNSET-ADA-UNSET %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-adaptive-stripping=false" %s | FileCheck --check-prefix=CHECK-AGG-UNSET-ADA-OFF %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-adaptive-stripping=true" %s | FileCheck --check-prefix=CHECK-AGG-UNSET-ADA-ON %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-qdq-optimization-aggressive=false" %s | FileCheck --check-prefix=CHECK-AGG-OFF-ADA-UNSET %s
// RUN: vpux-opt --init-compiler="platform=%platform% enable-qdq-optimization-aggressive=true" %s | FileCheck --check-prefix=CHECK-AGG-ON-ADA-UNSET %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// CHECK-AGG-OFF-ADA-OFF: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-OFF-ADA-ON: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-ON-ADA-OFF: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-ON-ADA-ON: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-UNSET-ADA-UNSET: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-UNSET-ADA-OFF: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-UNSET-ADA-ON: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-OFF-ADA-UNSET: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
// CHECK-AGG-ON-ADA-UNSET: module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping
module @CheckQDQOptimizationAggressiveEnablesAdaptiveStripping {
    // CHECK-DAG:    {{  }}config.PipelineOptions @Options {

    // CHECK-AGG-OFF-ADA-OFF-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : false
    // CHECK-AGG-OFF-ADA-OFF-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : false

    // CHECK-AGG-OFF-ADA-ON-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : false
    // CHECK-AGG-OFF-ADA-ON-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true

    // CHECK-AGG-ON-ADA-OFF-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : true
    // CHECK-AGG-ON-ADA-OFF-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true

    // CHECK-AGG-ON-ADA-ON-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : true
    // CHECK-AGG-ON-ADA-ON-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true

    // CHECK-AGG-UNSET-ADA-UNSET-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : false
    // CHECK-AGG-UNSET-ADA-UNSET-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true

    // CHECK-AGG-UNSET-ADA-OFF-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : false
    // CHECK-AGG-UNSET-ADA-OFF-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : false

    // CHECK-AGG-UNSET-ADA-ON-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : false
    // CHECK-AGG-UNSET-ADA-ON-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true

    // CHECK-AGG-OFF-ADA-UNSET-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : false
    // CHECK-AGG-OFF-ADA-UNSET-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true

    // CHECK-AGG-ON-ADA-UNSET-DAG:    {{    }}config.Option @config.EnableQDQOptimizationAggressive : true
    // CHECK-AGG-ON-ADA-UNSET-DAG:    {{    }}config.Option @config.EnableAdaptiveStripping : true
}
