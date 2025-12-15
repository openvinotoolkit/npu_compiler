//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-LABEL: @SingleNesting
module @SingleNesting {
    module @SubModule {
        func.func private @foo(%arg: tensor<f32>) -> tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = Core.NestedCall @SubModule::@foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }
}

// -----

// CHECK-LABEL: @DoubleNesting
module @DoubleNesting {
    module @Sub1 {
        module @Sub2 {
            func.func private @foo(%arg: tensor<f32>) -> tensor<f32>
        }
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = Core.NestedCall @Sub1::@Sub2::@foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }
}

// -----

module @NotAFuncOp {
    module @SubModule {
        module @foo {
        }
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        // expected-error@+1 {{'Core.NestedCall' op @SubModule::@foo does not point to a valid 'func.func' op}}
        %0 = Core.NestedCall @SubModule::@foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }
}

// -----

module @IsolatedFromAbove {
    module @Sub1 {
        func.func private @foo(%arg: tensor<f32>) -> tensor<f32>
    }

    module @Sub2 {
        func.func @main(%arg: tensor<f32>) -> tensor<f32> {
            // expected-error@+1 {{'Core.NestedCall' op @Sub1::@foo does not point to a valid 'func.func' op}}
            %0 = Core.NestedCall @Sub1::@foo(%arg) : (tensor<f32>) -> tensor<f32>
            return %0: tensor<f32>
        }
    }
}

// -----

module @OperandTypesMismatch {
    module @Sub1 {
        func.func private @foo(%arg: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        // expected-error@+1 {{'Core.NestedCall' op @Sub1::@foo operand types do not match}}
        %0 = Core.NestedCall @Sub1::@foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }
}

// -----

module @ResultTypesMismatch {
    module @Sub1 {
        func.func private @foo(%arg: tensor<f32>) -> (tensor<f32>, tensor<f32>)
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        // expected-error@+1 {{'Core.NestedCall' op @Sub1::@foo result types do not match}}
        %0 = Core.NestedCall @Sub1::@foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }
}

// -----

module @NotANestedSymbol {
    func.func private @foo(%arg: tensor<f32>) -> (tensor<f32>, tensor<f32>)

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        // expected-error@+1 {{'Core.NestedCall' op 'callee' must be a nested symbol}}
        %0 = Core.NestedCall @foo(%arg) : (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }
}
