//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --pack-nested-modules --verify-diagnostics %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: module @FunctionToPackBasic
module @FunctionToPackBasic {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func nested @helper_fn(%arg: tensor<f32>) -> tensor<f32> attributes {config.functionToPack = "helper_module"} {
        return %arg : tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @helper_fn(%arg) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
    }

    // CHECK: module @helper_module attributes {{.*}}config.packedModule{{.*}}
    // CHECK:     func.func nested @helper_fn

    // CHECK: func.func @main
    // CHECK:     Core.NestedCall @helper_module::@helper_fn
}

// -----

// CHECK-LABEL: module @FunctionToPackWithEntryPoint
module @FunctionToPackWithEntryPoint {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func nested @helper_fn(%arg: tensor<f32>) -> tensor<f32>
        attributes {config.functionToPack = "my_module"} {
        return %arg : tensor<f32>
    }

    func.func @entry_fn(%arg: tensor<f32>) -> tensor<f32>
        attributes {config.functionToPack = "my_module", config.functionToPackEntryPoint} {
        %0 = call @helper_fn(%arg) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @entry_fn(%arg) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
    }

    // CHECK: module @my_module attributes {{.*}}config.packedModule{{.*}}
    // CHECK:     net.NetworkInfo entryPoint : @entry_fn
    // CHECK:     func.func nested @helper_fn
    // CHECK:     func.func @entry_fn

    // CHECK: func.func @main
    // CHECK:     Core.NestedCall @my_module::@entry_fn
}

// -----

// CHECK-LABEL: module @MultipleFunctionToPackModules
module @MultipleFunctionToPackModules {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func nested @fn_a(%arg: tensor<f32>) -> tensor<f32>
        attributes {config.functionToPack = "module_a", config.functionToPackEntryPoint} {
        return %arg : tensor<f32>
    }

    func.func nested @fn_b(%arg: tensor<f32>) -> tensor<f32>
        attributes {config.functionToPack = "module_b"} {
        return %arg : tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @fn_a(%arg) : (tensor<f32>) -> tensor<f32>
        %1 = call @fn_b(%0) : (tensor<f32>) -> tensor<f32>
        return %1 : tensor<f32>
    }

    // CHECK: module @module_b attributes {{.*}}config.packedModule{{.*}}
    // CHECK:     net.NetworkInfo entryPoint : @fn_b
    // CHECK: module @module_a attributes {{.*}}config.packedModule{{.*}}
    // CHECK:     net.NetworkInfo entryPoint : @fn_a

    // CHECK: func.func @main
    // CHECK:     Core.NestedCall @module_a::@fn_a
    // CHECK:     Core.NestedCall @module_b::@fn_b
}

// -----

// Test that functions inside nested modules with FunctionToPack attribute are not processed.
// The pass only processes direct children of the top module.
// Correctness is verified by checking that @foo remains inside @existingModule with its
// original config.functionToPack attribute, and that the call still references @existingModule::@foo.

// CHECK-LABEL: module @NestedModuleFunctionToPack
module @NestedModuleFunctionToPack {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    module @existingModule {
        func.func nested @foo(%arg: tensor<f32>) -> tensor<f32>
            attributes {config.functionToPack = "targetModule"} {
            return %arg : tensor<f32>
        }
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = Core.NestedCall @existingModule::@foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
    }

    // CHECK: module @existingModule
    // CHECK:     func.func nested @foo{{.*}}config.functionToPack = "targetModule"

    // CHECK: func.func @main
    // CHECK:     Core.NestedCall @existingModule::@foo
}
