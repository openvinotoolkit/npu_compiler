//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --platform=%platform% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --host-compile %s | FileCheck %s --check-prefixes=CHECK,CHECK-%platform%
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x2560x1000xf16>
        DataInfo "input2" : tensor<1x16x2560x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x2560x1000xf16>
    }

    // CHECK:                 module [[MODULE0:@.+]] attributes {
    // CHECK-SAME:              config.compilationMode = #config.compilation_mode<HostCompile>
    // CHECK-NPU4000:         builtin.module @ReservedMemory
    // CHECK-NPU4000:         module @DmaProfilingReservedMemory

    // CHECK-NPU4000:           func.func [[FUNC0:@.+]]([[_:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16, {{.*}}>, [[_:%.+]]: memref<1x[[STEP]]x1000x16xf16, {{.*}}>, [[_:%.+]]: memref<1x[[STEP]]x1000x16xf16, {{.*}}>) -> memref<1x[[STEP]]x1000x16xf16, {{.*}}> {
    // CHECK-NPU5010:           func.func [[FUNC0:@.+]]([[_:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16, {{.*}}>, [[_:%.+]]: memref<1x[[STEP]]x1000x16xf16, {{.*}}>, [[_:%.+]]: memref<1x[[STEP]]x1000x16xf16, {{.*}}>) -> memref<1x[[STEP]]x1000x16xf16, {{.*}}> {

    // CHECK-NPU4000-COUNT-28:    VPURT.Task
    // CHECK-NPU5010-COUNT-20:    VPURT.Task
    // CHECK-NOT: IE.Add
    func.func @main(%arg0: tensor<1x16x2560x1000xf16, {order = #NHWC}>,
                    %arg1: tensor<1x16x2560x1000xf16, {order = #NHWC}>)
          -> tensor<1x16x2560x1000xf16, {order = #NHWC}> {
        %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x2560x1000xf16, {order = #NHWC}>,
            tensor<1x16x2560x1000xf16, {order = #NHWC}>
                -> tensor<1x16x2560x1000xf16, {order = #NHWC}>
        return %0 : tensor<1x16x2560x1000xf16, {order = #NHWC}>

        // CHECK:               func.func @main([[ARG0:%.+]]: memref<1x2560x1000x16xf16>, [[ARG1:%.+]]: memref<1x2560x1000x16xf16>, [[ARG2:%.+]]: memref<1x2560x1000x16xf16>) attributes {{{.*}}HostExec.HostCompileInferenceExec{{.*}}}

        // CHECK-DAG:             [[C0:%.+]] = arith.constant 0 : index
        // CHECK-NPU4000-DAG:     [[END:%.+]] = arith.constant 2560 : index
        // CHECK-NPU5010-DAG:     [[END:%.+]] = arith.constant 2560 : index
        // CHECK-DAG:             [[STEP_VAR:%.+]] = arith.constant [[STEP]] : index

        // CHECK:                 [[GROUP:%.+]] = async.create_group
        // CHECK:                 scf.for [[ARG3:%.+]] = [[C0]] to [[END]] step [[STEP_VAR]] {

        // CHECK:                   [[POS_WITH_BACKTRACK:%.+]] = affine.min

        // CHECK-NPU4000:           [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]
        // CHECK-NPU4000:           [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]
        // CHECK-NPU5010:           [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]
        // CHECK-NPU5010:           [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]

        // CHECK:                   [[CAST0:%.+]] = memref.cast [[SUBVIEW0]]
        // CHECK:                   [[CAST1:%.+]] = memref.cast [[SUBVIEW1]]

        // CHECK-NPU4000:           [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]
        // CHECK-NPU5010:           [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]

        // CHECK:                   [[CAST2:%.+]] = memref.cast [[SUBVIEW2]]
        // CHECK:                   [[TOKEN:%.+]] = async.execute
        // CHECK:                       Core.NestedCall [[MODULE0]]::[[FUNC0]]([[CAST0]], [[CAST1]], [[CAST2]])
        // CHECK:                       async.yield
        // CHECK:                   async.add_to_group [[TOKEN]], [[GROUP]]

        // CHECK:                 async.await_all [[GROUP]]
        // CHECK:                 return
    }
}
