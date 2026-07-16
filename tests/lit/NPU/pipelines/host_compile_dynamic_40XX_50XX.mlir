//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --host-compile %s | FileCheck %s --check-prefixes=CHECK,CHECK-%platform%
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU5020

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

    // CHECK: func.func @output_shape([[ARG0:%.+]]: memref<1x?x1000x16xf16>, [[ARG1:%.+]]: memref<1x?x1000x16xf16>, [[ARG2:%.+]]: memref<4xi64>) attributes {[[ANY_ATTR:.+]]} {
    // CHECK:    [[CST_3:%.+]] = arith.constant 3 : index
    // CHECK:    [[CST_0:%.+]] = arith.constant 0 : index
    // CHECK:    [[CST_1000_i64:%.+]] = arith.constant 1000 : i64
    // CHECK:    [[CST_2:%.+]] = arith.constant 2 : index
    // CHECK:    [[CST_16_i64:%.+]] = arith.constant 16 : i64
    // CHECK:    [[CST_1_i64:%.+]] = arith.constant 1 : i64
    // CHECK:    [[CST_1:%.+]] = arith.constant 1 : index
    // CHECK:    [[DIM:%.+]] = memref.dim [[ARG0]], [[CST_1]] : memref<1x?x1000x16xf16>
    // CHECK:    [[IDX_CAST:%.+]] = arith.index_cast [[DIM]] : index to i64
    // CHECK:    memref.store [[CST_1_i64]], [[ARG2]][[[CST_0]]] : memref<4xi64>
    // CHECK:    memref.store [[CST_16_i64]], [[ARG2]][[[CST_1]]] : memref<4xi64>
    // CHECK:    memref.store [[IDX_CAST]], [[ARG2]][[[CST_2]]] : memref<4xi64>
    // CHECK:    memref.store [[CST_1000_i64]], [[ARG2]][[[CST_3]]] : memref<4xi64>
    // CHECK:    return
    // }

    // CHECK: func.func @main_func0_static([[func0_ARG0:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>, [[func0_ARG1:%.+]]: memref<1x[[STEP]]x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>, [[func0_ARG2:%.+]]: memref<1x[[STEP]]x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>>) -> memref<1x[[STEP]]x1000x16xf16, strided<[?, ?, ?, ?], offset: ?>> {

    // CHECK-NPU4000-COUNT-6: VPUIP.NCEClusterTask
    // CHECK-NPU5010-COUNT-3: VPUIP.NCEClusterTask
    // CHECK-NPU5020-COUNT-1: VPUIP.NCEClusterTask
    // CHECK-NOT: IE.Add

    // With auto unrolling enabled (default for HostCompile pipeline), the host main function
    // contains a cascade of scf.for loops: the first loop calls the largest merged kernel
    // (auto-unrolled), followed by progressively smaller merged kernels (cascaded unrolling),
    // and a final residual loop calling the base single-tile kernel (main_func0_static).

    // CHECK: func.func @main([[ARG0:%.+]]: memref<1x?x1000x16xf16>, [[ARG1:%.+]]: memref<1x?x1000x16xf16>, [[ARG2:%.+]]: memref<1x?x1000x16xf16>) attributes {[[ANY_ATTR:.+]]} {
    // CHECK: [[BASE_STEP:%.+]] = arith.constant [[STEP]] : index
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[C1:%.+]] = arith.constant 1 : index
    // CHECK: [[DIM:%.+]] = memref.dim [[ARG0]], [[C1]] : memref<1x?x1000x16xf16>
    // CHECK: [[CMP_SGE_OUTER:%.+]] = arith.cmpi sge, [[DIM]], [[BASE_STEP]] : index
    // CHECK: cf.assert [[CMP_SGE_OUTER]], "Not enough elements to backtrack in scf.for loop for Output tensor"
    // First merged loop — largest unroll factor (auto-derived), calls merged_vpu_func_0.
    // CHECK: scf.for {{%.+}} = [[C0]] to {{%.+}} step {{%.+}} {
    // CHECK:   Core.NestedCall @{{.+}}::@merged_vpu_func_0(
    // CHECK: }
    // Residual loop — single-tile base kernel, iterates from cascade end to [[DIM]].
    // CHECK: scf.for {{%.+}} = {{%.+}} to [[DIM]] step [[BASE_STEP]] {
    // CHECK:   Core.NestedCall @{{.+}}::@main_func0_static(
    // CHECK: }
    // CHECK: async.await_all
    // CHECK: return
}
