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

    // CHECK: func.func @main_func0_static([[func0_ARG0:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16, @DDR>, [[func0_ARG1:%.+]]: memref<1x[[STEP]]x1000x16xf16, @DDR>, [[func0_ARG2:%.+]]: memref<1x[[STEP]]x1000x16xf16, @DDR>)
    // CHECK-SAME: -> memref<1x[[STEP]]x1000x16xf16, @DDR> {

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
    // CHECK: scf.for [[ARG3:%.+]] = [[C0]] to [[DIM]] step [[STEP_VAR]] {
    // CHECK: [[MIN:%.+]] = affine.min #map([[ARG3]])
    // CHECK: [[CMP:%.+]] = arith.cmpi ne, [[MIN]], [[STEP_VAR]] : index
    // CHECK: [[IF:%.+]] = scf.if [[CMP]] -> (index) {
    // CHECK: [[SUB1:%.+]] = arith.subi [[STEP_VAR]], [[MIN]] : index
    // CHECK: [[CMP1:%.+]] = arith.cmpi sgt, [[ARG3]], [[SUB1]] : index
    // CHECK: cf.assert [[CMP1]], "Not enough elements to backtrack in scf.for loop"
    // CHECK: [[SUB2:%.+]] = arith.subi [[ARG3]], [[SUB1]] : index
    // CHECK: scf.yield [[SUB2]] : index
    // CHECK: } else {
    // CHECK: scf.yield [[ARG3]] : index
    // CHECK: }

    // CHECK: [[SUBVIEW:%.+]] = memref.subview [[ARG0]][0, [[IF]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK: [[SUBVIEW_0:%.+]] = memref.subview [[ARG1]][0, [[IF]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK: [[CAST:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK: [[CAST_0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW_0]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK: [[SUBVIEW_1:%.+]] = memref.subview [[ARG2]][0, [[IF]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1] : memref<1x?x1000x16xf16> to memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>>
    // CHECK: [[CAST_1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW_1]] : memref<1x[[STEP]]x1000x16xf16, strided<[?, 16000, 16, 1], offset: ?>> to memref<1x[[STEP]]x1000x16xf16>
    // CHECK: [[TOKEN:%.+]], [[BODYRESULTS:%.+]] = async.execute -> !async.value<memref<1x[[STEP]]x1000x16xf16>> {
    // CHECK: [[RESULT:%.+]] = Core.NestedCall @Module0::@main_func0_static([[CAST]], [[CAST_0]], [[CAST_1]]) : (memref<1x[[STEP]]x1000x16xf16>, memref<1x[[STEP]]x1000x16xf16>, memref<1x[[STEP]]x1000x16xf16>) -> memref<1x[[STEP]]x1000x16xf16>
    // CHECK: async.yield [[RESULT]] : memref<1x[[STEP]]x1000x16xf16>
    // CHECK: }
    // CHECK: [[ADD_TO_GROUP_RES:%.+]] = async.add_to_group [[TOKEN]], [[GROUP]] : !async.token
    // CHECK: [[AWAIT:%.+]] = async.await [[BODYRESULTS]] : !async.value<memref<1x[[STEP]]x1000x16xf16>>
    // CHECK: }
    // CHECK: async.await_all [[GROUP]]
    // CHECK: return [[ARG2]] : memref<1x?x1000x16xf16>
}
