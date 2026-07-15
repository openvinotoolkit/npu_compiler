//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --unpack-nested-modules %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: @UnpackPackedModule
module @UnpackPackedModule {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @my_helper_module attributes {config.packedModule} {
        func.func nested @helper_fn(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }
    // CHECK-NOT: module @my_helper_module

    // CHECK: func.func nested @helper_fn([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   return [[ARG]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %result = Core.NestedCall @my_helper_module::@helper_fn(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %result : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[RESULT:%.+]] = call @helper_fn([[ARG]])
    // CHECK:   return [[RESULT]]
}

// -----

// CHECK-LABEL: @UnpackOnlyPackedModules
module @UnpackOnlyPackedModules {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @regular_module {
        func.func nested @regular_fn(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }
    // CHECK: module @regular_module

    module @packed_module attributes {config.packedModule} {
        func.func nested @packed_fn(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }
    // CHECK-NOT: module @packed_module

    // CHECK: func.func nested @packed_fn([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   return [[ARG]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %r1 = Core.NestedCall @packed_module::@packed_fn(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %r2 = Core.NestedCall @regular_module::@regular_fn(%r1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %r2 : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[R1:%.+]] = call @packed_fn([[ARG]])
    // CHECK:   [[R2:%.+]] = Core.NestedCall @regular_module::@regular_fn([[R1]])
    // CHECK:   return [[R2]]
}

// -----

// CHECK-LABEL: @UnpackMultiplePackedModules
module @UnpackMultiplePackedModules {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input": tensor<2x2xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf32>
    }

    module @module_a attributes {config.packedModule} {
        net.NetworkInfo entryPoint : @fn_a1 inputsInfo : {
            DataInfo "in_0": tensor<2x2xf32>
        } outputsInfo : {
            DataInfo "out_0" : tensor<2x2xf32>
        }

        func.func @fn_a1(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            %0 = func.call @fn_a2(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
        }
        func.func nested @fn_a2(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }
    // CHECK-NOT: module @module_a

    module @module_b attributes {config.packedModule} {
        func.func nested @fn_b(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
            return %arg : tensor<2x2xf32>
        }
    }
    // CHECK-NOT: module @module_b

    // CHECK: func.func nested @fn_a1([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[R:%.+]] = call @fn_a2([[ARG]])
    // CHECK:   return [[R]]

    // CHECK: func.func nested @fn_a2([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   return [[ARG]]

    // CHECK: func.func nested @fn_b([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   return [[ARG]]

    func.func @main(%arg: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %r1 = Core.NestedCall @module_a::@fn_a1(%arg) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        %r2 = Core.NestedCall @module_b::@fn_b(%r1) : (tensor<2x2xf32>) -> tensor<2x2xf32>
        return %r2 : tensor<2x2xf32>
    }

    // CHECK: func.func @main([[ARG:%.+]]: tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK:   [[R1:%.+]] = call @fn_a1([[ARG]])
    // CHECK:   [[R2:%.+]] = call @fn_b([[R1]])
    // CHECK:   return [[R2]]
}
