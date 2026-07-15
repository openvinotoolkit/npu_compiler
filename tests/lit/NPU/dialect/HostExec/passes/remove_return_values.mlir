//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --platform=%platform% --prepare-host-function-for-async-execution="remove-return-values=true" %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @RemoveReturnValues attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x256x?xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x16x256x?xf16>
    }

    module @Module0 {
        func.func nested @main_func0(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
                          -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
            return %arg0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        }
    }

    func.func @main(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> attributes {HostExec.HostCompileInferenceExec} {
        %0 = Core.NestedCall @Module0::@main_func0(%arg0)
               : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    }

    // CHECK-LABEL: @RemoveReturnValues

    // CHECK: func.func @main({{%.+}}: tensor<{{.+}}>)
    // CHECK-NOT: -> tensor
    // CHECK-SAME: attributes {HostExec.HostCompileInferenceExec}

    // CHECK: return{{$}}
}

// -----

module @InferenceExecCalledFromAnotherFunc attributes {config.compilationMode = #config.compilation_mode<HostCompile>} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x16x?x?xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x?x?xf16>
    }
    module @Module0 {
        func.func nested @main_func0(%arg0: memref<1x16x256x240xf16>, %arg1: memref<1x16x256x240xf16>) -> memref<1x16x256x240xf16> {
            return %arg1 : memref<1x16x256x240xf16>
        }
    }

    func.func @main(%arg0: memref<1x16x?x?xf16>, %arg1: memref<1x16x?x?xf16>) -> memref<1x16x?x?xf16> attributes {HostExec.HostCompileInferenceExec, config.pureHostCompileFunc} {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c256 = arith.constant 256 : index
        %dim_h = memref.dim %arg0, %c2 : memref<1x16x?x?xf16>
        scf.for %arg2 = %c0 to %dim_h step %c256 {
            %subview_0 = memref.subview %arg0[0, 0, %arg2, 0] [1, 16, 256, 240] [1, 1, 1, 1]
                                      : memref<1x16x?x?xf16> to memref<1x16x256x240xf16, strided<[?, ?, ?, 1], offset: ?>>
            %cast_0 = builtin.unrealized_conversion_cast %subview_0
                                      : memref<1x16x256x240xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x256x240xf16>
            %subview_1 = memref.subview %arg1[0, 0, %arg2, 0] [1, 16, 256, 240] [1, 1, 1, 1]
                                      : memref<1x16x?x?xf16> to memref<1x16x256x240xf16, strided<[?, ?, ?, 1], offset: ?>>
            %cast_1 = builtin.unrealized_conversion_cast %subview_1
                                      : memref<1x16x256x240xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x256x240xf16>
            Core.NestedCall @Module0::@main_func0(%cast_0, %cast_1)
                                      : (memref<1x16x256x240xf16>, memref<1x16x256x240xf16>) -> memref<1x16x256x240xf16>
        }
        return %arg1 : memref<1x16x?x?xf16>
    }

    func.func @caller(%arg0: memref<1x16x?x?xf16>, %arg1: memref<1x16x?x?xf16>) -> memref<1x16x?x?xf16> attributes {config.pureHostCompileFunc} {
        %result = func.call @main(%arg0, %arg1) : (memref<1x16x?x?xf16>, memref<1x16x?x?xf16>) -> memref<1x16x?x?xf16>
        return %result : memref<1x16x?x?xf16>
    }

    // CHECK-LABEL: @InferenceExecCalledFromAnotherFunc

    // @Module0::@main_func0 is not modified by the pass — it is a nested kernel, not a host function.
    // CHECK: func.func nested @main_func0([[MF0_ARG0:%.+]]: memref<1x16x256x240xf16>, [[MF0_ARG1:%.+]]: memref<1x16x256x240xf16>)
    // CHECK-SAME: -> memref<1x16x256x240xf16>

    // @main: return value is stripped; Core.NestedCall is wrapped in async.execute inside an async.group.
    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x16x?x?xf16>, [[ARG1:%.+]]: memref<1x16x?x?xf16>)
    // CHECK-NOT: -> memref
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[C256:%.+]] = arith.constant 256 : index
    // CHECK: [[DIM_H:%.+]] = memref.dim [[ARG0]]
    // CHECK: [[SUB:%.+]] = arith.subi [[DIM_H]], [[C0]]
    // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[C256]]
    // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]] : !async.group
    // CHECK: scf.for [[ARG2:%.+]] = [[C0]] to [[DIM_H]] step [[C256]] {
    // CHECK: [[TOKEN:%.+]] = async.execute {
    // CHECK: Core.NestedCall @Module0::@main_func0(%{{.+}}, %{{.+}}) : (memref<1x16x256x240xf16>, memref<1x16x256x240xf16>) -> memref<1x16x256x240xf16>
    // CHECK: async.add_to_group [[TOKEN]], [[GROUP]] : !async.token
    // CHECK: async.await_all [[GROUP]]
    // CHECK: return{{$}}

    // @caller: no async wrapping; call @main has void result type; return value is stripped.
    // CHECK-LABEL: func.func @caller
    // CHECK-SAME: ([[C_ARG0:%.+]]: memref<1x16x?x?xf16>, [[C_ARG1:%.+]]: memref<1x16x?x?xf16>)
    // CHECK-NOT: -> memref
    // CHECK-NOT: async.execute
    // CHECK: call @main([[C_ARG0]], [[C_ARG1]]) : (memref<1x16x?x?xf16>, memref<1x16x?x?xf16>) -> ()
    // CHECK-NOT: async.execute
    // CHECK: return{{$}}
}
