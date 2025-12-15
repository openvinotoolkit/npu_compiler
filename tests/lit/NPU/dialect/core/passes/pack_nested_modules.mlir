//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --pack-nested-modules %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// Note: This test simulates a common use-case related to weight separation.
module @InitAndWrapper {
    net.NetworkInfo entryPoint : @main_wrapper inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func private @init(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<f32>)

    func.func private @main(%input: tensor<f32>, %cst_1: tensor<f32>, %cst_2: tensor<f32>) -> tensor<f32> attributes {do_not_nest}

    func.func @main_wrapper(%arg: tensor<f32>) -> tensor<f32> {
        %cst_1 = const.Declare tensor<f32> = dense<1.0> : tensor<f32>
        %cst_2 = const.Declare tensor<f32> = dense<2.0> : tensor<f32>
        %init:2 = call @init(%cst_1, %cst_2) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
        %main = call @main(%arg, %init#0, %init#1) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
        return %main: tensor<f32>
    }

    // Note: Check this for verify insertion location
    // CHECK: net.NetworkInfo


    // CHECK: module @Module0 {
    // CHECK:     func.func private @init
    // CHECK: }


    // CHECK: func.func private @main


    // CHECK: func.func @main_wrapper
    // CHECK:     [[CST_0:%.+]] = const.Declare tensor<f32> = dense<1.000000e+00> : tensor<f32>
    // CHECK:     [[CST_1:%.+]] = const.Declare tensor<f32> = dense<2.000000e+00> : tensor<f32>

    // Note: Only calls to nested functions are replaced by Core.NestedCall ops
    // CHECK:     Core.NestedCall @Module0::@init([[CST_0]], [[CST_1]]) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)

    // CHECK:     [[MAIN:%.+]] = call @main
    // CHECK:     return [[MAIN]]
}

// -----

module @MultipleSubModules {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    // Note: Check this for verify insertion location
    // CHECK: net.NetworkInfo

    // Note: private here is just set so that we don't have to define a function body

    // Cluster 1
    func.func private @foo_cluster1(tensor<f32>) -> tensor<f32>

    // Cluster 2
    func.func private @bar_cluster2(tensor<f32>) -> tensor<f32>

    func.func @goo_cluster2(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @bar_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    func.func @baz_cluster2(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @bar_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK: module @Module0
    // CHECK:     func.func private @f

    // CHECK: module @Module1 {
    // CHECK:     func.func private @bar_cluster2

    // CHECK:     func.func @goo_cluster2
    // CHECK:         call @bar_cluster2

    // CHECK:     func.func @baz_cluster2
    // CHECK:         call @bar_cluster2

    // Top Cluster (no nesting)
    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @foo_cluster1(%arg): (tensor<f32>) -> tensor<f32>
        %1 = call @goo_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        %2 = call @baz_cluster2(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK: func.func @main
    // CHECK:     Core.NestedCall @Module0::@foo_cluster1
    // CHECK:     Core.NestedCall @Module1::@goo_cluster2
    // CHECK:     Core.NestedCall @Module1::@baz_cluster2
}

// -----

module @NoNesting {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<f32>
    } outputsInfo : {
        DataInfo "output" : tensor<f32>
    }

    func.func private @foo(%arg: tensor<f32>) -> tensor<f32> attributes {do_not_nest}

    func.func @bar(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @foo(%arg): (tensor<f32>) -> tensor<f32>
        return %0 : tensor<f32>
    }

    func.func @main(%arg: tensor<f32>) -> tensor<f32> {
        %0 = call @bar(%arg): (tensor<f32>) -> tensor<f32>
        return %0: tensor<f32>
    }

    // CHECK:     module @NoNesting
    // CHECK-NOT: module
    // CHECK:     func.func private @foo

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

    func.func private @foo(%arg0: memref<f16, @DDR>, %arg1: memref<f16, @DDR>) -> memref<f16, @DDR>

    // CHECK: module @Module0 {
    // CHECK:     func.func private @foo
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
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%netIn : memref<f16, @DDR>)
                outputs(%inAlloc : memref<f16, @DDR>)
                -> memref<f16, @DDR>
        }
        VPURT.Task waits(%b_fooCall1CopyIn : !VPURT.Barrier) updates(%b_fooCall1 : !VPURT.Barrier) {
            %0 = func.call @foo(%inAlloc, %outAlloc)
                : (memref<f16, @DDR>, memref<f16, @DDR>) -> memref<f16, @DDR>
            // CHECK: Core.NestedCall @Module0::@foo
        }

        VPURT.Task waits(%b_fooCall1 : !VPURT.Barrier) {
            %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%outAlloc : memref<f16, @DDR>)
                outputs(%netOut : memref<f16, @DDR>)
                -> memref<f16, @DDR>
        }
        return %arg1 : memref<f16, @DDR>
    }
}
