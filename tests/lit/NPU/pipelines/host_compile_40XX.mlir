//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --host-compile %s | FileCheck %s --check-prefixes=CHECK,CHECK-%arch%
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @CopyInputOutput
module @CopyInputOutput {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" : tensor<1x3x60x60xf16>
  } outputsInfo : {
    DataInfo "output" : tensor<1x3x60x60xf16>
  }

// CHECK-NPU40XX: builtin.module @ReservedMemory

  func.func private @main_part1(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %0 = VPU.Copy(%arg0) : tensor<1x3x60x60xf16> -> tensor<1x3x60x60xf16>
    return %0 : tensor<1x3x60x60xf16>
  }

  func.func @main(%arg0: tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16> {
    %0 = call @main_part1(%arg0) : (tensor<1x3x60x60xf16>) -> tensor<1x3x60x60xf16>
    return %0 : tensor<1x3x60x60xf16>
  }
  // CHECK:             module [[MODULE0:@.+]] attributes {
  // CHECK-SAME:          config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
  // CHECK-NPU40XX:     module @DmaProfilingReservedMemory

  // CHECK:             func.func private [[FUNC0:@.+]]([[_:%.+]]: memref<1x3x60x60xf16>, [[_:%.+]]: memref<1x3x60x60xf16>)
  // CHECK-SAME:          -> memref<1x3x60x60xf16> {
  // CHECK-COUNT-1:       VPURT.Task
  // CHECK-NOT:         VPU.Copy

  // CHECK:             func.func @main([[ARG0:%.+]]: memref<1x3x60x60xf16>, [[ARG1:%.+]]: memref<1x3x60x60xf16>) -> memref<1x3x60x60xf16> {
  // CHECK:               [[C0:%.+]] = arith.constant 0 : index
  // CHECK:               [[ALLOC:%.+]] = memref.alloc() {alignment = 64 : i64} : memref<21600xi8>
  // CHECK:               [[VIEW:%.+]] = memref.view [[ALLOC]][[[C0]]][] : memref<21600xi8> to memref<1x3x60x60xf16>

  // CHECK:               [[TOKEN:%.+]], [[BODY_RESULTS:%.+]] = async.execute
  // CHECK:                   [[OUT0:%.+]] = Core.NestedCall [[MODULE0]]::[[FUNC0]]([[ARG0]], [[VIEW]])
  // CHECK:                   async.yield [[OUT0]]
  // CHECK:               [[RESULT:%.+]] = async.await [[BODY_RESULTS]]

  // CHECK:               memref.copy [[RESULT]], [[ARG1]] : memref<1x3x60x60xf16> to memref<1x3x60x60xf16>
  // CHECK:               return [[ARG1]] : memref<1x3x60x60xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @StaticEltwiseNHWC
module @StaticEltwiseNHWC {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x2560x1000xf16>
        DataInfo "input2" : tensor<1x16x2560x1000xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x2560x1000xf16>
    }

    // CHECK-NPU40XX:         builtin.module @ReservedMemory

    // CHECK:                 module [[MODULE0:@.+]] attributes {
    // CHECK-SAME:              config.compilationMode = #config.compilation_mode<HostCompile>, config.revisionID = #config.revision_id<REVISION_NONE>} {
    // CHECK-NPU40XX:         module @DmaProfilingReservedMemory

    // CHECK:                   func.func [[FUNC0:@.+]]([[_:%.+]]: memref<1x[[STEP:.+]]x1000x16xf16>, [[_:%.+]]: memref<1x[[STEP]]x1000x16xf16>, [[_:%.+]]: memref<1x[[STEP]]x1000x16xf16>) -> memref<1x[[STEP]]x1000x16xf16> {

    // CHECK-NPU40XX-COUNT-27:    VPURT.Task
    // CHECK-NOT: IE.Add
    func.func @main(%arg0: tensor<1x16x2560x1000xf16, {order = #NHWC}>,
                    %arg1: tensor<1x16x2560x1000xf16, {order = #NHWC}>)
          -> tensor<1x16x2560x1000xf16, {order = #NHWC}> {
        %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} :
            tensor<1x16x2560x1000xf16, {order = #NHWC}>,
            tensor<1x16x2560x1000xf16, {order = #NHWC}>
                -> tensor<1x16x2560x1000xf16, {order = #NHWC}>
        return %0 : tensor<1x16x2560x1000xf16, {order = #NHWC}>

        // CHECK:               func.func @main([[ARG0:%.+]]: memref<1x2560x1000x16xf16>, [[ARG1:%.+]]: memref<1x2560x1000x16xf16>, [[ARG2:%.+]]: memref<1x2560x1000x16xf16>) -> memref<1x2560x1000x16xf16>

        // CHECK-DAG:             [[C0:%.+]] = arith.constant 0 : index
        // CHECK-DAG:             [[END:%.+]] = arith.constant 2560 : index
        // CHECK-DAG:             [[STEP_VAR:%.+]] = arith.constant [[STEP]] : index

        // CHECK:                 [[GROUP:%.+]] = async.create_group
        // CHECK:                 scf.for [[ARG3:%.+]] = [[C0]] to [[END]] step [[STEP_VAR]] {

        // CHECK:                   [[POS_WITH_BACKTRACK:%.+]] = scf.if

        // CHECK:                   [[SUBVIEW0:%.+]] = memref.subview [[ARG0]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]
        // CHECK:                   [[SUBVIEW1:%.+]] = memref.subview [[ARG1]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]


        // CHECK:                   [[CAST0:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW0]]
        // CHECK:                   [[CAST1:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW1]]

        // CHECK:                   [[SUBVIEW2:%.+]] = memref.subview [[ARG2]][0, [[POS_WITH_BACKTRACK]], 0, 0] [1, [[STEP]], 1000, 16] [1, 1, 1, 1]

        // CHECK:                   [[CAST2:%.+]] = builtin.unrealized_conversion_cast [[SUBVIEW2]]
        // CHECK:                   [[TOKEN:%.+]], [[BODY_RESULTS:%.+]] = async.execute
        // CHECK:                       [[CALL_RES:%.+]] = Core.NestedCall [[MODULE0]]::[[FUNC0]]([[CAST0]], [[CAST1]], [[CAST2]])
        // CHECK:                       async.yield [[CALL_RES]]
        // CHECK:                   async.add_to_group [[TOKEN]], [[GROUP]]
        // CHECK:                   [[RESULT:%.+]] = async.await [[BODY_RESULTS]]

        // CHECK:                 async.await_all [[GROUP]]
        // CHECK:                 return [[ARG2]] : memref<1x2560x1000x16xf16>
    }
}
