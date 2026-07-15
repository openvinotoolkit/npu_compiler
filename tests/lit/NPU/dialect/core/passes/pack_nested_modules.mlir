//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --pack-nested-modules %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Note: This test simulates a common use-case related to weight separation.
module @InitAndWrapper {
    net.NetworkInfo entryPoint : @main_wrapper inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func nested @init(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>)
    {
        return %arg0, %arg1 : tensor<f32>, tensor<f32>
    }

    func.func nested @main(%input: tensor<f32>, %cst_1: tensor<f32>, %cst_2: tensor<f32>) -> tensor<f32> attributes {do_not_nest}

    func.func @main_wrapper(%arg: tensor<f32>) -> tensor<f32> {
        %cst_1 = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
        %cst_2 = const.Declare tensor<f32> = dense<2.0> : tensor<f32>
        %init:2 = call @init(%cst_1, %cst_2) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
        %main = call @main(%arg, %init#0, %init#1) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
        return %main: tensor<f32>
    }

    // Note: Check this for verify insertion location
    // CHECK: net.NetworkInfo entryPoint : @main_wrapper inputsInfo
    // CHECK:   DataInfo "input" : tensor<f32>
    // CHECK: } outputsInfo : {
    // CHECK:   DataInfo "output" : tensor<f32>


    // CHECK-LABEL: module @Module0 attributes {{.+}} {
    // CHECK:     config.PipelineOptions @Options {
    // CHECK:     config.Resources {{.+}} of @NCE at {{.+}} MHz
    // CHECK:     config.Resources {{.+}} of @global
    // CHECK:     net.NetworkInfo entryPoint : @init inputsInfo
    // CHECK:       DataInfo "in_0" tensorNames = ["in_0"] : tensor<f32>
    // CHECK:       DataInfo "in_1" tensorNames = ["in_1"] : tensor<f32>
    // CHECK:     } outputsInfo : {
    // CHECK:       DataInfo "out_0" tensorNames = ["out_0"] : tensor<f32>
    // CHECK:       DataInfo "out_1" tensorNames = ["out_1"] : tensor<f32>

    // CHECK:     func.func nested @init

    // CHECK: func.func nested @main(tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32> attributes {do_not_nest}
    // CHECK: func.func @main_wrapper
    // CHECK:     [[CST_0:%.+]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    // CHECK:     [[CST_1:%.+]] = const.Declare tensor<f32> = dense<2.000000e+00> : tensor<f32>

    // Note: Only calls to nested functions are replaced by Core.NestedCall ops
    // CHECK:     Core.NestedCall @Module0::@init([[CST_0]], [[CST_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)

    // CHECK:     [[MAIN:%.+]] = call @main
    // CHECK:     return [[MAIN]]
}

// -----

// CHECK-LABEL: module @MultipleSubModules attributes {{.+}} {
module @MultipleSubModules {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func nested @foo_cluster1(%arg: tensor<f32>) -> tensor<f32> {
        return %arg: tensor<f32>
    }

    func.func nested @bar_cluster2(tensor<f32> ) -> tensor<f32>

    func.func @goo_cluster2(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @bar_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    func.func @baz_cluster2(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @bar_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @foo_cluster1(%arg): (tensor<f32>) -> tensor<f32>
        %1 = call @goo_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        %2 = call @baz_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK-LABEL: module @Module0 attributes {{.+}} {
    // CHECK:     config.PipelineOptions @Options {
    // CHECK:     config.Resources {{.+}} of @NCE at {{.+}} MHz
    // CHECK:     config.Resources {{.+}} of @global
    // CHECK:     net.NetworkInfo entryPoint : @foo_cluster1 inputsInfo : {
    // CHECK:       DataInfo "in_0" tensorNames = ["in_0"] : tensor<f32>
    // CHECK:     } outputsInfo : {
    // CHECK:       DataInfo "out_0" tensorNames = ["out_0"] : tensor<f32>
    // CHECK:     func.func nested @foo_cluster1
    // CHECK: }
    // CHECK-LABEL: module @Module1 attributes {{.+}} {
    // CHECK-NOT:    config.PipelineOptions @Options
    // CHECK-NOT:    config.Resources {{.+}} of @NCE at {{.+}} MHz
    // CHECK-NOT:    config.Resources {{.+}} of @global
    // CHECK-NOT:    net.NetworkInfo entryPoint
    // CHECK:  func.func nested @bar_cluster2(tensor<f32>) -> tensor<f32>
    // CHECK:  func.func @goo_cluster2
    // CHECK:  func.func @baz_cluster2

    // CHECK: func.func @main([[ARG:%.+]]: tensor<f32>) -> tensor<f32> {
    // CHECK:   [[foo:%.+]] = Core.NestedCall @Module0::@foo_cluster1([[ARG]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   [[goo:%.+]] = Core.NestedCall @Module1::@goo_cluster2([[ARG]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   [[baz:%.+]] = Core.NestedCall @Module1::@baz_cluster2([[ARG]]) : (tensor<f32>) -> tensor<f32>
    // CHECK:   return [[foo]] : tensor<f32>

}

// -----

module @NoNesting {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func nested @foo(%arg: tensor<f32>) -> tensor<f32> attributes {do_not_nest}

    func.func @bar(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @foo(%arg): (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @bar(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK: module @NoNesting
    // CHECK-NOT: module
    // CHECK:     func.func nested @foo

    // CHECK:     func.func @bar
    // CHECK:         call @foo

    // CHECK:     func.func @main
    // CHECK:         call @bar
}

// -----

module @VPUIP {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
      DataInfo "input" : tensor<f16>
    } outputsInfo : {
      DataInfo "output" : tensor<f16>
    }

    func.func nested @foo(%arg0: memref<f16, @DDR>, %arg1: memref<f16, @DDR>) -> memref<f16, @DDR>
    {
        return %arg1 : memref<f16, @DDR>
    }

    // CHECK-LABEL: module @Module0 attributes {{.+}} {
    // CHECK:     config.PipelineOptions @Options {
    // CHECK:     config.Resources {{.+}} of @NCE at {{.+}} MHz
    // CHECK:     config.Resources {{.+}} of @global
    // CHECK:     net.NetworkInfo entryPoint : @foo inputsInfo : {
    // CHECK:       DataInfo "in_0" tensorNames = ["in_0"] : tensor<f16>
    // CHECK:       DataInfo "in_1" tensorNames = ["in_1"] : tensor<f16>
    // CHECK:     } outputsInfo : {
    // CHECK:       DataInfo "out_0" tensorNames = ["out_0"] : tensor<f16>
    // CHECK:     func.func nested @foo
    // CHECK: }

    func.func @main(%arg0: memref<f16, @DDR>, %arg1: memref<f16, @DDR>) -> memref<f16, @DDR> {
    // CHECK: func.func @main
        %netIn = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<f16, @DDR>
        %netOut = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<f16, @DDR>

        %inAlloc = VPURT.DeclareBuffer <DDR> <0> -> memref<f16, @DDR>
        %outAlloc = VPURT.DeclareBuffer <DDR> <24576> -> memref<f16, @DDR>
        %b_fooCall1CopyIn = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %b_fooCall1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

        VPURT.Task updates(%b_fooCall1CopyIn : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%netIn : memref<f16, @DDR>)
                outputs(%inAlloc : memref<f16, @DDR>)
                -> memref<f16, @DDR>
        }
        VPURT.Task waits(%b_fooCall1CopyIn : !VPURT.Barrier) updates(%b_fooCall1 : !VPURT.Barrier) {
            %0 = func.call @foo(%inAlloc, %outAlloc)
                : (memref<f16, @DDR>, memref<f16, @DDR>) -> memref<f16, @DDR>
            // CHECK: Core.NestedCall @Module0::@foo
        }

        VPURT.Task waits(%b_fooCall1 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA <{port = 0 : i64}> inputs(%outAlloc : memref<f16, @DDR>)
                outputs(%netOut : memref<f16, @DDR>)
                -> memref<f16, @DDR>
        }
        return %arg1 : memref<f16, @DDR>
    }
}
