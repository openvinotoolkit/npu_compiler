//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --host-compile %s | FileCheck %s --check-prefixes=CHECK,CHECK-%arch%

// REQUIRES: arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @EltwiseNHWCDynamic
module @EltwiseNHWCDynamic {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x?x1000xf16>
        DataInfo "input2" : tensor<1x16x?x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x?x1000xf16>
    }

    func.func @main(%arg0: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>) -> tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}> {
      %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>,
            tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>
                -> tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>
      return %0 : tensor<1x16x?x1000xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 2560, 1000]> : tensor<4xsi64>, order = #NHWC}>
    }

    // CHECK: func.func @main_func0_static([[func0_ARG0:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16>, [[func0_ARG1:%.+]]: memref<1x[[STEP]]x1000x16xf16>, [[func0_ARG2:%.+]]: memref<1x[[STEP]]x1000x16xf16>) -> memref<1x[[STEP]]x1000x16xf16> {

    // CHECK-NPU40XX-COUNT-6: VPUIP.NCEClusterTask
    // CHECK-NOT: IE.Add

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x?x1000x16xf16>, [[ARG1:%.+]]: memref<1x?x1000x16xf16>, [[ARG2:%.+]]: memref<1x?x1000x16xf16>) -> memref<1x?x1000x16xf16> {
    // CHECK: [[STEP_VAR:%.+]] = arith.constant [[STEP]] : index
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[C1:%.+]] = arith.constant 1 : index
    // CHECK: [[DIM:%.+]] = memref.dim [[ARG0]], [[C1]] : memref<1x?x1000x16xf16>
    // CHECK: [[SUB:%.+]] = arith.subi [[DIM]], [[C0]] : index
    // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[STEP_VAR]] : index
    // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]] : !async.group
    // CHECK: scf.for [[IF:%.+]] = [[C0]] to [[DIM]] step [[STEP_VAR]] {
    // CHECK:   [[MIN:%.+]] = affine.min #{{.+}}([[IF]]){{\[}}[[DIM]]{{\]}}
    // CHECK:   [[CMP_EQ:%.+]] = arith.cmpi eq, [[IF]], [[C0]] : index
    // CHECK:   [[IF_RESULT:%.+]] = scf.if [[CMP_EQ]] -> (index) {
    // CHECK:     [[CMP_SGE:%.+]] = arith.cmpi sge, [[MIN]], [[STEP_VAR]] : index
    // CHECK:     cf.assert [[CMP_SGE]], "Not enough elements to backtrack in scf.for loop"
    // CHECK:     scf.yield [[IF]] : index
    // CHECK:   } else {
    // CHECK:     [[ADDI:%.+]] = arith.addi [[IF]], [[STEP_VAR]] : index
    // CHECK:     [[CMP_SLT:%.+]] = arith.cmpi slt, [[ADDI]], [[DIM]] : index
    // CHECK:     [[IF_RESULT_2:%.+]] = scf.if [[CMP_SLT]] -> (index) {
    // CHECK:       scf.yield [[IF]] : index
    // CHECK:     } else {
    // CHECK:       [[CMP_EQ_2:%.+]] = arith.cmpi eq, [[ADDI]], [[DIM]] : index
    // CHECK:       [[IF_RESULT_3:%.+]] = scf.if [[CMP_EQ_2]] -> (index) {
    // CHECK:         scf.yield [[IF]] : index
    // CHECK:       } else {
    // CHECK:         [[APPLY:%.+]] = affine.apply #{{.+}}([[IF]]){{\[}}{{%.+}}{{\]}}
    // CHECK:         scf.yield [[APPLY]] : index
    // CHECK:       }
    // CHECK:       scf.yield [[IF_RESULT_3]] : index
    // CHECK:     }
    // CHECK:     scf.yield [[IF_RESULT_2]] : index
    // CHECK:   }
    // CHECK:   [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[IF_RESULT]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK:   [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[IF_RESULT]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK:   [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[IF_RESULT]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK:   [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   [[TOKEN:%.+]], [[BODY_RESULTS:%.+]] = async.execute -> !async.value<memref<1x[[STEP]]x1000x16xf16>> {
    // CHECK:     [[NESTED_CALL:%.+]] = Core.NestedCall @Module0::@main_func0_static([[CAST0]], [[CAST1]], [[CAST2]]) : (memref<1x[[STEP]]x1000x16xf16>, memref<1x[[STEP]]x1000x16xf16>, memref<1x[[STEP]]x1000x16xf16>) -> memref<1x[[STEP]]x1000x16xf16>
    // CHECK:     async.yield [[NESTED_CALL]] : memref<1x[[STEP]]x1000x16xf16>
    // CHECK:   }
    // CHECK:   [[ADD_TO_GROUP:%.+]] = async.add_to_group [[TOKEN]], [[GROUP]] : !async.token
    // CHECK:   [[AWAIT:%.+]] = async.await [[BODY_RESULTS]] : !async.value<memref<1x[[STEP]]x1000x16xf16>>
    // CHECK: }
    // CHECK: async.await_all [[GROUP]]
    // CHECK: return [[ARG2]] : memref<1x?x1000x16xf16>
}
