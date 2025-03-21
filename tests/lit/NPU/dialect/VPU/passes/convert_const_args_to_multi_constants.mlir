//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-const-args-to-multi-constants --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @MultipleFunctionsMultipleConstants
module @MultipleFunctionsMultipleConstants {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<48x48x3x3xf32>
    } outputsInfo : {
        DataInfo "output0" : tensor<48x48x3x3xf32>
    }

    // CHECK:        const.Data @Data {
    // CHECK-DAG:        const.Rodata [[RODATA_0:@.+]] dense<1.000000e+00> : tensor<48x48x3x3xf16>
    // CHECK-DAG:        const.Rodata [[RODATA_1:@.+]] dense<2.000000e+00> : tensor<48x48x3x3xf16>
    // CHECK:        }
    // CHECK:        const.BundleData @BundleData {
    // CHECK-DAG:        const.RodataBundle [[BUNDLE_0:@.+]] = [@Data::[[RODATA_0]], @Data::[[RODATA_0]]] : tensor<48x48x3x3xf16>
    // CHECK-DAG:        const.RodataBundle [[BUNDLE_1:@.+]] = [@Data::[[RODATA_0]], @Data::[[RODATA_1]]] : tensor<48x48x3x3xf16>
    // CHECK-DAG:        const.RodataBundle [[BUNDLE_2:@.+]] = [@Data::[[RODATA_1]], @Data::[[RODATA_1]]] : tensor<48x48x3x3xf16>
    // CHECK-DAG:        const.RodataBundle [[BUNDLE_3:@.+]] = [@Data::[[RODATA_0]], @Data::[[RODATA_1]]] : tensor<48x48x3x3xf16>
    // CHECK:        }

    func.func private @main_fn1(%arg0: tensor<48x48x3x3xf32>, %arg1: tensor<48x48x3x3xf32>, %arg2: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
        %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        %1 = VPU.Add(%arg0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        %2 = VPU.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        return %2 : tensor<48x48x3x3xf32>
    }

    // CHECK:        func.func private @main_fn1([[ARG0:%.+]]: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
    // CHECK-DAG:        [[MCST_0:%.+]] = const.MultiDeclare tensor<48x48x3x3xf32> = @BundleData::[[BUNDLE_0]] : tensor<48x48x3x3xf16>, [#const.Add<5.000000e+00 : f64>, #const.CastElemType<f32>]
    // CHECK-DAG:        [[MCST_1:%.+]] = const.MultiDeclare tensor<48x48x3x3xf32> = @BundleData::[[BUNDLE_1]] : tensor<48x48x3x3xf16>, [#const.Add<5.000000e+00 : f64>, #const.CastElemType<f32>]
    // CHECK:            [[R0:%.+]] = VPU.Add([[ARG0]], [[MCST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:            [[R1:%.+]] = VPU.Add([[ARG0]], [[MCST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:            [[R2:%.+]] = VPU.Add([[R0]], [[R1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:            return [[R2]] : tensor<48x48x3x3xf32>
    // CHECK:        }

    func.func private @main_fn2(%arg0: tensor<48x48x3x3xf32>, %arg1: tensor<48x48x3x3xf32>, %arg2: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
        %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        %1 = VPU.Add(%arg0, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        %2 = VPU.Add(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        return %2 : tensor<48x48x3x3xf32>
    }

    // CHECK:        func.func private @main_fn2([[ARG0:%.+]]: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
    // CHECK-DAG:        [[MCST_1:%.+]] = const.MultiDeclare tensor<48x48x3x3xf32> = @BundleData::[[BUNDLE_2]] : tensor<48x48x3x3xf16>, [#const.Add<5.000000e+00 : f64>, #const.CastElemType<f32>]
    // CHECK-DAG:        [[MCST_2:%.+]] = const.MultiDeclare tensor<48x48x3x3xf32> = @BundleData::[[BUNDLE_3]] : tensor<48x48x3x3xf16>, [#const.Add<5.000000e+00 : f64>, #const.CastElemType<f32>]
    // CHECK:            [[R0:%.+]] = VPU.Add([[ARG0]], [[MCST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:            [[R1:%.+]] = VPU.Add([[ARG0]], [[MCST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:            [[R2:%.+]] = VPU.Add([[R0]], [[R1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
    // CHECK:            return [[R2]] : tensor<48x48x3x3xf32>
    // CHECK:        }

    func.func @main(%input: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
        %cst_weights1 = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf16>, [#const.Add<5.0>, #const.CastElemType<f32>]
        %cst_weights2 = const.Declare tensor<48x48x3x3xf32> = dense<2.0> : tensor<48x48x3x3xf16>, [#const.Add<5.0>, #const.CastElemType<f32>]
        %call1 = call @main_fn1(%input, %cst_weights1, %cst_weights1) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
        %call2 = call @main_fn1(%call1, %cst_weights1, %cst_weights2) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
        %call3 = call @main_fn2(%call2, %cst_weights2, %cst_weights1) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
        %call4 = call @main_fn2(%call3, %cst_weights2, %cst_weights2) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
        return %call4 : tensor<48x48x3x3xf32>
    }

    // CHECK:        func.func @main([[ARG:%.+]]: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
    // CHECK:            [[R0:%.+]] = call @main_fn1([[ARG]]) : (tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    // CHECK:            [[R1:%.+]] = call @main_fn1([[R0]]) : (tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    // CHECK:            [[R2:%.+]] = call @main_fn2([[R1]]) : (tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    // CHECK:            [[R3:%.+]] = call @main_fn2([[R2]]) : (tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    // CHECK:            return [[R3]] : tensor<48x48x3x3xf32>
    // CHECK:        }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// expected-error@+1 {{IR contains unexpected op 'const.Data'}}
module @ContainsDataOp {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<4xf32>
    } outputsInfo : {
        DataInfo "output0" : tensor<4xf32>
    }

    const.Data @Data {
    }

    func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
        return %input : tensor<4xf32>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// expected-error@+1 {{IR contains unexpected op 'const.BundleData'}}
module @ContainsBundleDataOp {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<4xf32>
    } outputsInfo : {
        DataInfo "output0" : tensor<4xf32>
    }

    const.BundleData @BundleData {
    }

    func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
        return %input : tensor<4xf32>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @NestedCalls {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<4xf32>
    } outputsInfo : {
        DataInfo "output0" : tensor<4xf32>
    }

    func.func private @main_fn1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        return %arg0 : tensor<4xf32>
    }

    // expected-error@+1 {{'func.func' op main_fn2 contains disallowed 'func.Call' op outside of net func}}
    func.func private @main_fn2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        %0 = call @main_fn1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
        return %0 : tensor<4xf32>
    }

    func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
        %cst_weights1 = const.Declare tensor<4xf32> = dense<1.0> : tensor<4xf32>
        %call1 = call @main_fn2(%cst_weights1) : (tensor<4xf32>) -> tensor<4xf32>
        return %call1 : tensor<4xf32>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// expected-error@+1 {{A possible bundle would contain 'const.Declare' ops with differing base content types or transformations for 'func.Func' main_fn1. This is unexpected and might indicate a problem with outlining!}}
module @DifferingTransformations {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<4xf32>
    } outputsInfo : {
        DataInfo "output0" : tensor<4xf32>
    }

    func.func private @main_fn1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        return %arg0 : tensor<4xf32>
    }

    func.func @main(%input: tensor<4xf32>) -> tensor<4xf32> {
        %cst_weights1 = const.Declare tensor<4xf32> = dense<1.0> : tensor<4xf32>
        %cst_weights2 = const.Declare tensor<4xf32> = dense<1.0> : tensor<4xf32>, [#const.Add<1.0>]
        %call1 = call @main_fn1(%cst_weights1) : (tensor<4xf32>) -> tensor<4xf32>
        %call2 = call @main_fn1(%cst_weights2) : (tensor<4xf32>) -> tensor<4xf32>
        return %call1 : tensor<4xf32>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @CandidateSharedWithNonConstant
module @CandidateSharedWithNonConstant {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<48x48x3x3xf32>
    } outputsInfo : {
        DataInfo "output0" : tensor<48x48x3x3xf32>
    }

    // CHECK:        const.Data @Data {
    // CHECK:        }
    // CHECK:        const.BundleData @BundleData {
    // CHECK:        }

    func.func private @main_fn1(%arg0: tensor<48x48x3x3xf32>, %arg1: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
        %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        %2 = VPU.Add(%0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32> -> tensor<48x48x3x3xf32>
        return %2 : tensor<48x48x3x3xf32>
    }

    func.func @main(%input: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
        %cst_weights1 = const.Declare tensor<48x48x3x3xf32> = dense<1.0> : tensor<48x48x3x3xf16>, [#const.Add<5.0>, #const.CastElemType<f32>]
        %cst_weights2 = const.Declare tensor<48x48x3x3xf32> = dense<2.0> : tensor<48x48x3x3xf16>, [#const.Add<5.0>, #const.CastElemType<f32>]
        %call1 = call @main_fn1(%cst_weights1, %cst_weights1) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
        %call2 = call @main_fn1(%cst_weights2, %input) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
        return %call1 : tensor<48x48x3x3xf32>
    }

    // CHECK:        func.func @main([[ARG:%.+]]: tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32> {
    // CHECK:            [[CST_0:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<1.000000e+00> : tensor<48x48x3x3xf16>, [#const.Add<5.000000e+00 : f64>, #const.CastElemType<f32>]
    // CHECK:            [[CST_1:%.+]] = const.Declare tensor<48x48x3x3xf32> = dense<2.000000e+00> : tensor<48x48x3x3xf16>, [#const.Add<5.000000e+00 : f64>, #const.CastElemType<f32>]
    // CHECK:            [[R0:%.+]] = call @main_fn1([[CST_0]], [[CST_0]]) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    // CHECK:            [[R1:%.+]] = call @main_fn1([[CST_1]], [[ARG]]) : (tensor<48x48x3x3xf32>, tensor<48x48x3x3xf32>) -> tensor<48x48x3x3xf32>
    // CHECK:            return [[R0]] : tensor<48x48x3x3xf32>
    // CHECK:        }
}
