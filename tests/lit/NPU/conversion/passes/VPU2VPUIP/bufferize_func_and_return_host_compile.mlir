//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=HostCompile" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

module @BufferizeHost {
    func.func nested @main_func0(%arg0: tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16> {
        return %arg0 : tensor<1x90x1000x16xf16>
    }
    func.func @main(%arg0: tensor<1x720x1000x16xf16>) -> tensor<1x720x1000x16xf16> attributes {config.pureHostCompileFunc} {
        %c90 = arith.constant 90 : index
        %c720 = arith.constant 720 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<1x720x1000x16xf16>
        %1 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %0) -> (tensor<1x720x1000x16xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                             : tensor<1x720x1000x16xf16> to tensor<1x90x1000x16xf16>
            %2 = func.call @main_func0(%extracted_slice) : (tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16>
            %inserted_slice = tensor.insert_slice %2 into %arg3[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                            : tensor<1x90x1000x16xf16> into tensor<1x720x1000x16xf16>
            scf.yield %inserted_slice : tensor<1x720x1000x16xf16>
        }
        return %1 : tensor<1x720x1000x16xf16>
    }
// CHECK: func.func nested @main_func0([[_:%.+]]: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16>
// CHECK:   [[C90:%.+]] = arith.constant 90 : index
// CHECK:   [[C720:%.+]] = arith.constant 720 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc()
// CHECK:   [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C720]] step [[C90]] iter_args([[ARG3:%.+]] = [[ALLOC]])
// CHECK:       [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:       [[CAST:%.+]] = memref.cast [[SUBVIEW]]
// CHECK:                    : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:       [[CALL:%.+]] = func.call @main_func0([[CAST]])
// CHECK:       [[INSERTED:%.+]] = memref.subview [[ARG3]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:       memref.copy [[CALL]], [[INSERTED]]
// CHECK:            : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
// CHECK:       scf.yield [[ARG3]] : memref<1x720x1000x16xf16>

// CHECK:   return [[FOR]] : memref<1x720x1000x16xf16>
}

// -----

module @BufferizeHostNestedPureHostCompileFunction {
    func.func nested @main_func0_pure_host_compile(%arg0: tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16> attributes {HostExec.HostCompileInferenceExec, HostExec.InferenceExpectedCommandListsNumber = 1 : i64, config.functionToPackEntryPoint, config.pureHostCompileFunc} {
        return %arg0 : tensor<1x90x1000x16xf16>
    }
    func.func @main(%arg0: tensor<1x720x1000x16xf16>) -> tensor<1x720x1000x16xf16> attributes {config.pureHostCompileFunc} {
        %c90 = arith.constant 90 : index
        %c720 = arith.constant 720 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<1x720x1000x16xf16>
        %1 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %0) -> (tensor<1x720x1000x16xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                             : tensor<1x720x1000x16xf16> to tensor<1x90x1000x16xf16>
            %2 = func.call @main_func0_pure_host_compile(%extracted_slice) : (tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16>
            %inserted_slice = tensor.insert_slice %2 into %arg3[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                            : tensor<1x90x1000x16xf16> into tensor<1x720x1000x16xf16>
            scf.yield %inserted_slice : tensor<1x720x1000x16xf16>
        }
        return %1 : tensor<1x720x1000x16xf16>
    }
// CHECK: func.func nested @main_func0_pure_host_compile([[_:%.+]]: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>> attributes {[[ALL:.+]]}

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16>
// CHECK:   [[C90:%.+]] = arith.constant 90 : index
// CHECK:   [[C720:%.+]] = arith.constant 720 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc()
// CHECK:   [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C720]] step [[C90]] iter_args([[ARG3:%.+]] = [[ALLOC]])
// CHECK:       [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:       [[CAST:%.+]] = memref.cast [[SUBVIEW]]
// CHECK:                    : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:       [[CALL:%.+]] = func.call @main_func0_pure_host_compile([[CAST]])
// CHECK:       [[INSERTED:%.+]] = memref.subview [[ARG3]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:       memref.copy [[CALL]], [[INSERTED]]
// CHECK:            : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
// CHECK:       scf.yield [[ARG3]] : memref<1x720x1000x16xf16>

// CHECK:   return [[FOR]] : memref<1x720x1000x16xf16>
}


// -----

module @BufferizeHost {
    module @Module {
        func.func nested @main_func0(%arg0: tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16> {
            return %arg0 : tensor<1x90x1000x16xf16>
        }
    }

    func.func @main(%arg0: tensor<1x720x1000x16xf16>) -> tensor<1x720x1000x16xf16> attributes {config.pureHostCompileFunc} {
        %c90 = arith.constant 90 : index
        %c720 = arith.constant 720 : index
        %c0 = arith.constant 0 : index
        %0 = tensor.empty() : tensor<1x720x1000x16xf16>
        %1 = scf.for %arg2 = %c0 to %c720 step %c90 iter_args(%arg3 = %0) -> (tensor<1x720x1000x16xf16>) {
            %extracted_slice = tensor.extract_slice %arg0[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                             : tensor<1x720x1000x16xf16> to tensor<1x90x1000x16xf16>
            %2 = Core.NestedCall @Module::@main_func0(%extracted_slice) : (tensor<1x90x1000x16xf16>) -> tensor<1x90x1000x16xf16>
            %inserted_slice = tensor.insert_slice %2 into %arg3[0, %arg2, 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
                            : tensor<1x90x1000x16xf16> into tensor<1x720x1000x16xf16>
            scf.yield %inserted_slice : tensor<1x720x1000x16xf16>
        }
        return %1 : tensor<1x720x1000x16xf16>
    }
// CHECK: func.func nested @main_func0([[ARG_0:%[^:]+]]: memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>)
// CHECK-SAME: -> memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x720x1000x16xf16>) -> memref<1x720x1000x16xf16>
// CHECK:   [[C90:%.+]] = arith.constant 90 : index
// CHECK:   [[C720:%.+]] = arith.constant 720 : index
// CHECK:   [[C0:%.+]] = arith.constant 0 : index
// CHECK:   [[ALLOC:%.+]] = memref.alloc()
// CHECK:   [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[C720]] step [[C90]] iter_args([[ARG3:%.+]] = [[ALLOC]])
// CHECK:       [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:       [[CAST:%.+]] = memref.cast [[SUBVIEW]]
// CHECK:                    : memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>> to memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:       [[CALL:%.+]] = Core.NestedCall @Module::@main_func0([[CAST]])
// CHECK:       [[INSERTED:%.+]] = memref.subview [[ARG3]][0, [[ARG2]], 0, 0] [1, 90, 1000, 16] [1, 1, 1, 1]
// CHECK:       memref.copy [[CALL]], [[INSERTED]]
// CHECK:            : memref<1x90x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>> to memref<1x90x1000x16xf16, strided<[11520000, 16000, 16, 1], offset: ?>>
// CHECK:       scf.yield [[ARG3]] : memref<1x720x1000x16xf16>

// CHECK:   return [[FOR]] : memref<1x720x1000x16xf16>
}

// -----
module @BufferizeReinterpretCast {
    module @Module0 {
        func.func @kernel(%arg0: tensor<1x100x100x16xf16>) -> tensor<1x100x100x16xf16>  {
            %result = Core.ReinterpretCast(%arg0) : tensor<1x100x100x16xf16> -> tensor<1x100x100x16xf16>
            return %result : tensor<1x100x100x16xf16>
        }
    }

    func.func @main(%arg0: tensor<1x?x?x16xf16>) -> tensor<1x?x?x16xf16> attributes {config.pureHostCompileFunc} {
        %subview = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 100, 100, 16] [1, 1, 1, 1]
                 : tensor<1x?x?x16xf16> to tensor<1x100x100x16xf16>
        %result = Core.NestedCall @Module0::@kernel(%subview) : (tensor<1x100x100x16xf16>) -> tensor<1x100x100x16xf16>
        return %arg0 : tensor<1x?x?x16xf16>
    }
// CHECK: func.func @kernel([[ARG0:%.+]]: memref<1x100x100x16xf16, strided<[?, ?, ?, ?], offset: ?>>)
// CHECK-SAME: -> memref<1x100x100x16xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:   [[REINTERPRET:%.+]] = Core.ReinterpretCast([[ARG0]])
// CHECK:                       : memref<1x100x100x16xf16, strided<[?, ?, ?, ?], offset: ?>> -> memref<1x100x100x16xf16>
// CHECK:   [[CAST:%.+]] = memref.cast [[REINTERPRET]]
// CHECK:                : memref<1x100x100x16xf16> to memref<1x100x100x16xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:   return [[CAST]] : memref<1x100x100x16xf16, strided<[?, ?, ?, ?], offset: ?>>

// CHECK: func.func @main([[ARG0:%.+]]: memref<1x?x?x16xf16>) -> memref<1x?x?x16xf16>
// CHECK:   [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, 0, 0, 0] [1, 100, 100, 16] [1, 1, 1, 1]
// CHECK:                   : memref<1x?x?x16xf16> to memref<1x100x100x16xf16, strided<[?, ?, 16, 1]>>
// CHECK:   [[CAST:%.+]] = memref.cast [[SUBVIEW]]
// CHECK:                : memref<1x100x100x16xf16, strided<[?, ?, 16, 1]>> to memref<1x100x100x16xf16, strided<[?, ?, ?, ?], offset: ?>>
// CHECK:   [[CALL:%.+]] = Core.NestedCall @Module0::@kernel([[CAST]])
// CHECK:   return [[ARG0]] : memref<1x?x?x16xf16>
}
