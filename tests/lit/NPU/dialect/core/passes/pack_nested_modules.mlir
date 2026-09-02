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

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @BoundsOrderRestore {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<?x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548, 1]> : tensor<4xsi64>, order = #NCHW}>
    } outputsInfo : {
        DataInfo "output" : tensor<?x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548, 1]> : tensor<4xsi64>, order = #NCHW}>
    }

    func.func nested @main_func1(%arg0: tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16> {
        %bounded = Core.ReinterpretCast(%arg0) : tensor<?x1x?x1xf16> -> tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 548, 1]> : tensor<4xsi64>, order = #NCHW}>
        %restored = Core.ReinterpretCast(%bounded) : tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 548, 1]> : tensor<4xsi64>, order = #NCHW}> -> tensor<?x1x?x1xf16>
        return %restored : tensor<?x1x?x1xf16>
    }

    func.func @main_func0(%arg0: tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16> {
        return %arg0 : tensor<?x1x?x1xf16>
    }

    func.func @main_func2_static(%arg0: tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16> {
        return %arg0 : tensor<?x1x?x1xf16>
    }

    func.func @main_func3_static(%arg0: tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16> {
        return %arg0 : tensor<?x1x?x1xf16>
    }

    func.func @main(%arg0: tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16> {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index

        scf.for %i = %c0 to %c1 step %c1 {
            %pre = func.call @main_func0(%arg0) : (tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16>
            scf.yield
        }

        %call = func.call @main_func1(%arg0) : (tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16>

        scf.for %j = %c0 to %c1 step %c1 {
            %post = func.call @main_func2_static(%call) : (tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16>
            scf.yield
        }

        %tail = func.call @main_func3_static(%call) : (tensor<?x1x?x1xf16>) -> tensor<?x1x?x1xf16>
        return %tail : tensor<?x1x?x1xf16>
    }

    // CHECK-LABEL: module @Module0 attributes {{.+}} {
    // CHECK:     net.NetworkInfo entryPoint : @main_func1 inputsInfo : {
    // CHECK:       DataInfo "in_0" tensorNames = ["in_0"] : tensor<?x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     } outputsInfo : {
    // CHECK:       DataInfo "out_0" tensorNames = ["out_0"] : tensor<?x1x?x1xf16, {bounds = #const.OpaqueI64Elements<[32, 1, 548, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     func.func nested @main_func1
    // CHECK-LABEL: func.func @main
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @BoundsOrderRestoreOnOutputTensor {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input0" : tensor<1x?x?x1xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 640, 1]> : tensor<4xsi64>, order = #NCHW}>
        DataInfo "input1" : tensor<1x?x?x2xf16, {bounds = #const.OpaqueI64Elements<[1, 180, 320, 2]> : tensor<4xsi64>, order = #NCHW}>
        DataInfo "input2" : tensor<4xf32>
    } outputsInfo : {
        DataInfo "output" friendlyName = "output/sink_port_0" tensorNames = ["output"] : tensor<1x?x?x3xui8, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 3]> : tensor<4xsi64>, order = #NCHW}>
    }

    func.func @main_func2_static(%arg0: tensor<1x360x320x1xf16>, %arg1: tensor<1x180x160x2xf16>) -> tensor<1x360x320x3xf16> {
                %0 = VPU.Concat(%arg0, %arg0, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]} : tensor<1x360x320x1xf16>, tensor<1x360x320x1xf16>, tensor<1x360x320x1xf16> -> tensor<1x360x320x3xf16>
                return %0 : tensor<1x360x320x3xf16>
    }

    func.func nested @main_func1(%arg0: tensor<1x?x?x3xf16>, %arg1: tensor<4xf32>) -> tensor<1x?x?x3xf16> {
        %0 = Core.ReinterpretCast(%arg1) : tensor<4xf32> -> tensor<4xf32>
        %1 = Core.ReinterpretCast(%arg0) : tensor<1x?x?x3xf16> -> tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 640, 3]> : tensor<4xsi64>, order = #NCHW}>
        %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 640, 3]> : tensor<4xsi64>, order = #NCHW}> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 360, 640]> : tensor<4xsi64>, order = #NHWC}>
        %3 = VPU.Empty : tensor<1x1x1x1323060xui8>
        %4 = VPU.InterpolateDMA(%2, %0, %3) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>, nearest_mode = <FLOOR>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [0, 1, 2, 3], multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>} : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 360, 640]> : tensor<4xsi64>, order = #NHWC}>, tensor<4xf32>, tensor<1x1x1x1323060xui8> -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}>
        %5 = Core.ReinterpretCast(%4) : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 2160, 3840]> : tensor<4xsi64>, order = #NHWC}> -> tensor<1x?x?x3xf16>
        return %5 : tensor<1x?x?x3xf16>
    }

    func.func @main(%arg0: tensor<1x?x?x1xf16>, %arg1: tensor<1x?x?x2xf16>, %arg2: tensor<4xf32>) -> tensor<1x?x?x3xf16> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
        %c320 = arith.constant 320 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %dim = tensor.dim %arg0, %c1 : tensor<1x?x?x1xf16>
        %dim_0 = tensor.dim %arg0, %c2 : tensor<1x?x?x1xf16>
        %0 = tensor.empty(%dim, %dim_0) : tensor<1x?x?x3xf16>
        %3 = scf.for %arg3 = %c0 to %dim_0 step %c320 iter_args(%arg4 = %0) -> (tensor<1x?x?x3xf16>) {
            %33 = affine.min affine_map<(d0)[s0] -> (s0 - 320, d0)>(%arg3)[%dim_0]
            %34 = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%33)
            %extracted_slice = tensor.extract_slice %arg0[0, 0, %33, 0] [1, 360, 320, 1] [1, 1, 1, 1] : tensor<1x?x?x1xf16> to tensor<1x360x320x1xf16>
            %extracted_slice_8 = tensor.extract_slice %arg1[0, 0, %34, 0] [1, 180, 160, 2] [1, 1, 1, 1] : tensor<1x?x?x2xf16> to tensor<1x180x160x2xf16>
            %35 = func.call @main_func2_static(%extracted_slice, %extracted_slice_8) : (tensor<1x360x320x1xf16>, tensor<1x180x160x2xf16>) -> tensor<1x360x320x3xf16>
            %inserted_slice = tensor.insert_slice %35 into %arg4[0, 0, %33, 0] [1, 360, 320, 3] [1, 1, 1, 1] : tensor<1x360x320x3xf16> into tensor<1x?x?x3xf16>
            scf.yield %inserted_slice : tensor<1x?x?x3xf16>
        }
        %4 = call @main_func1(%3, %arg2) : (tensor<1x?x?x3xf16>, tensor<4xf32>) -> tensor<1x?x?x3xf16>
        return %4 : tensor<1x?x?x3xf16>
    }

    // CHECK-LABEL: module @Module1 attributes {{.+}} {
    // CHECK:     net.NetworkInfo entryPoint : @main_func1 inputsInfo : {
    // CHECK:       DataInfo "in_0" tensorNames = ["in_0"] : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 360, 640, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:       DataInfo "in_1" tensorNames = ["in_1"] : tensor<4xf32>
    // CHECK:     } outputsInfo : {
    // CHECK:       DataInfo "out_0" tensorNames = ["out_0"] : tensor<1x?x?x3xf16, {bounds = #const.OpaqueI64Elements<[1, 2160, 3840, 3]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:     }
    // CHECK:     func.func nested @main_func1
}
