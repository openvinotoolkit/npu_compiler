//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --prepare-host-function-for-async-execution %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0)[s0] -> (240, -d0 + s0)>
module @ControlFlowOutliningDynamicShape  {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<1x16x256x?xf16>
        DataInfo "input2" : tensor<1x16x256x?xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x16x256x?xf16>
    }
    module @Module0 {
        func.func private @main_func0(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                                      %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
                          -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
            %0 = VPU.NCE.Eltwise(%arg0, %arg1) {is_inplace = true, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                    op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>,
                    clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                    lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00],
                    fp_prelu_alpha = 1.000000e+00 : f64>}
                -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
            return %0 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        }
    }

    func.func @main(%arg0: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                    %arg1: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}> {
        %c3 = arith.constant 3 : index
        %dim = tensor.dim %arg0, %c3 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        %0 = tensor.empty(%dim) : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        %c0 = arith.constant 0 : index
        %c240 = arith.constant 240 : index
        %1 = scf.for %arg2 = %c0 to %dim step %c240 iter_args(%arg3 = %0) -> (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>) {
            %2 = affine.min #map(%arg2)[%dim]
            %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1]
                             : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
                             to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
            %extracted_slice_2 = tensor.extract_slice %arg1[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1]
                               : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
                               to tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

            %3 = Core.NestedCall @Module0::@main_func0(%extracted_slice, %extracted_slice_2)
               : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

            %inserted_slice = tensor.insert_slice %3 into %arg3[0, 0, 0, %arg2] [1, 16, 256, %2] [1, 1, 1, 1]
                            : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
                            into tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
            scf.yield %inserted_slice : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
        }

        %2 = Core.NestedCall @Module0::@main_func0(%1, %arg1)
               : (tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
                  tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
               -> tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>

        return %2 : tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>
    }
    // CHECK-LABEL: @ControlFlowOutliningDynamicShape

    // CHECK: module [[MODULE0:@.+]] {
    // CHECK: func.func private [[FUNC0:@.+]](%arg0: tensor<1x16x256x?xf16

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>,
    // CHECK-SAME: [[ARG1:%.+]]: tensor<1x16x256x?xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 256, 480]> : tensor<4xsi64>, order = #NHWC}>)
    // CHECK: [[C3:%.+]] = arith.constant 3 : index
    // CHECK: [[DIM:%.+]] = tensor.dim [[ARG0]], [[C3]]
    // CHECK: [[ALLOC:%.+]] = tensor.empty([[DIM]])
    // CHECK: [[C0:%.+]] = arith.constant 0 : index
    // CHECK: [[C240:%.+]] = arith.constant 240 : index
    // CHECK: [[SUB:%.+]] = arith.subi [[DIM]], [[C0]]
    // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[C240]]
    // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]]

    // CHECK: [[FOR:%.+]] = scf.for [[ARG2:%.+]] = [[C0]] to [[DIM]] step [[C240]] iter_args([[ARG3:%.+]] = [[ALLOC]])
    // CHECK: [[TOKEN:%.+]], [[RESULTS:%.+]] = async.execute
    // CHECK: Core.NestedCall [[MODULE0]]::[[FUNC0]]

    // CHECK: async.add_to_group [[TOKEN]], [[GROUP]] : !async.token
    // CHECK: async.await [[RESULTS]]

    // CHECK: scf.yield

    // CHECK: async.await_all [[GROUP]]

    // CHECK: [[TOKEN0:%.+]], [[RESULTS0:%.+]] = async.execute
    // CHECK: Core.NestedCall [[MODULE0]]::[[FUNC0]]
    // CHECK: [[RETURN:%.+]] = async.await [[RESULTS0]]

    // CHECK: return [[RETURN]]
}

// -----

    module @Module0 {
        func.func private @main_func0_dims_H_cases_0_static(%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
            return %arg1 : memref<1x16x28x1280xf16>
        }
    }

    module @Module1 {
        func.func private @main_func0_dims_H_cases_1_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
            return %arg1 : memref<1x16x28x1280xf16>
        }
    }

    module @Module2 {
        func.func private @main_func0_dims_H_cases_2_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
            return %arg1 : memref<1x16x28x1280xf16>
        }
    }

    func.func @AwaitAllCheckForLoop1D(%arg0: memref<1x16x?x1280xf16>, %arg1: memref<1x16x?x1280xf16>, %arg2: index, %arg3: index) -> memref<1x16x?x1280xf16> {
      %false = arith.constant false
      %c2 = arith.constant 2 : index
      %c28 = arith.constant 28 : index
      %c0 = arith.constant 0 : index
      %dim = memref.dim %arg0, %c2 : memref<1x16x?x1280xf16>
      scf.for %arg4 = %c0 to %dim step %c28 {
        %13 = scf.index_switch %arg2 -> memref<1x16x28x1280xf16>
        case 0 {
          %subview = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x30x1280xf16>
          %subview_0 = memref.subview %arg1[0, 0, %arg4, 0] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x28x1280xf16>
          %16 = Core.NestedCall @Module0::@main_func0_dims_H_cases_0_static(%14, %15) : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
          scf.yield %16 : memref<1x16x28x1280xf16>
        }
        case 1 {
          %subview = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x29x1280xf16>
          %subview_0 = memref.subview %arg1[0, 0, %arg4, 0] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x28x1280xf16>
          %16 = Core.NestedCall @Module1::@main_func0_dims_H_cases_1_static(%14, %15) : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
          scf.yield %16 : memref<1x16x28x1280xf16>
        }
        case 2 {
          %subview = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x29x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x29x1280xf16>
          %subview_0 = memref.subview %arg1[0, 0, %arg4, 0] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x28x1280xf16>
          %16 = Core.NestedCall @Module2::@main_func0_dims_H_cases_2_static(%14, %15) : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
          scf.yield %16 : memref<1x16x28x1280xf16>
        }
        default {
          cf.assert %false, "Unsupported case"
          %subview = memref.subview %arg0[0, 0, %arg3, 0] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x30x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x30x1280xf16>
          %subview_0 = memref.subview %arg1[0, 0, %arg4, 0] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x1280xf16> to memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>>
          %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, 1280, 1], offset: ?>> to memref<1x16x28x1280xf16>
          %16 = Core.NestedCall @Module0::@main_func0_dims_H_cases_0_static(%14, %15) : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
          scf.yield %16 : memref<1x16x28x1280xf16>
        }
      }
      return %arg1 : memref<1x16x?x1280xf16>

      // CHECK: module [[MODULE0:@.+]] {
      // CHECK: func.func private [[FUNC0:@.+]](%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
      // CHECK: module [[MODULE1:@.+]] {
      // CHECK: func.func private [[FUNC1:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
      // CHECK: module [[MODULE2:@.+]] {
      // CHECK: func.func private [[FUNC2:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)

      // CHECK: func.func @AwaitAllCheckForLoop1D([[ARG0:%.+]]: memref<1x16x?x1280xf16>, [[ARG1:%.+]]: memref<1x16x?x1280xf16>, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index)
      // CHECK: [[FALSE:%.+]] = arith.constant false
      // CHECK: [[C2:%.+]] = arith.constant 2 : index
      // CHECK: [[C28:%.+]] = arith.constant 28 : index
      // CHECK: [[C0:%.+]] = arith.constant 0 : index
      // CHECK: [[DIM:%.+]] = memref.dim [[ARG0]], [[C2]]
      // CHECK: [[SUB:%.+]] = arith.subi [[DIM]], [[C0]]
      // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[C28]]
      // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]]

      // CHECK: scf.for [[ARG4:%.+]] = [[C0]] to [[DIM]] step [[C28]] {

      // CHECK: scf.index_switch [[ARG2]]

      // CHECK: case 0 {
      // CHECK: async.add_to_group
      // CHECK-NOT: async.add_to_group
      // CHECK: async.await
      // CHECK-NOT: async.await

      // CHECK: case 1 {
      // CHECK: async.add_to_group
      // CHECK-NOT: async.add_to_group
      // CHECK: async.await
      // CHECK-NOT: async.await

      // CHECK: case 2 {
      // CHECK: async.add_to_group
      // CHECK-NOT: async.add_to_group
      // CHECK: async.await
      // CHECK-NOT: async.await

      // CHECK: default {
      // CHECK: cf.assert [[FALSE]], "Unsupported case"

      // CHECK: async.await_all [[GROUP]]
      // CHECK-NOT: }
      // CHECK-NOT: async.await_all
      // CHECK: return [[ARG1]]
    }

// -----

    module @Module0 {
        func.func private @main_func0_dims_H_cases_0_static(%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
            return %arg1 : memref<1x16x28x1280xf16>
        }
    }

    module @Module1 {
        func.func private @main_func0_dims_H_cases_1_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
            return %arg1 : memref<1x16x28x1280xf16>
        }
    }

    module @Module2 {
        func.func private @main_func0_dims_H_cases_2_static(%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16> {
            return %arg1 : memref<1x16x28x1280xf16>
        }
    }

    func.func @AwaitAllCheckForLoop2D(%arg0: memref<1x16x?x?xf16>, %arg1: memref<1x16x?x?xf16>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> memref<1x16x?x?xf16> {
      %false = arith.constant false
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c28 = arith.constant 28 : index
      %c1280 = arith.constant 1280 : index
      %c0 = arith.constant 0 : index
      %dim_h = memref.dim %arg0, %c2 : memref<1x16x?x?xf16>
      %dim_w = memref.dim %arg0, %c3 : memref<1x16x?x?xf16>
      scf.for %arg6 = %c0 to %dim_h step %c28 {
        scf.for %arg7 = %c0 to %dim_w step %c1280 {
          %100 = arith.shli %arg2, %c2 : index
          %101 = arith.ori %100, %arg4 : index
          %13 = scf.index_switch %101 -> memref<1x16x28x1280xf16>
          case 0 {
            %subview = memref.subview %arg0[0, 0, %arg3, %arg5] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x30x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x30x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x30x1280xf16>
            %subview_0 = memref.subview %arg1[0, 0, %arg6, %arg7] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x28x1280xf16>
            %16 = Core.NestedCall @Module0::@main_func0_dims_H_cases_0_static(%14, %15) : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
            scf.yield %16 : memref<1x16x28x1280xf16>
          }
          case 1 {
            %subview = memref.subview %arg0[0, 0, %arg3, %arg5] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x29x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x29x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x29x1280xf16>
            %subview_0 = memref.subview %arg1[0, 0, %arg6, %arg7] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x28x1280xf16>
            %16 = Core.NestedCall @Module1::@main_func0_dims_H_cases_1_static(%14, %15) : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
            scf.yield %16 : memref<1x16x28x1280xf16>
          }
          case 2 {
            %subview = memref.subview %arg0[0, 0, %arg3, %arg5] [1, 16, 29, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x29x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x29x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x29x1280xf16>
            %subview_0 = memref.subview %arg1[0, 0, %arg6, %arg7] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x28x1280xf16>
            %16 = Core.NestedCall @Module2::@main_func0_dims_H_cases_2_static(%14, %15) : (memref<1x16x29x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
            scf.yield %16 : memref<1x16x28x1280xf16>
          }
          default {
            cf.assert %false, "Unsupported case"
            %subview = memref.subview %arg0[0, 0, %arg3, %arg5] [1, 16, 30, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x30x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %14 = builtin.unrealized_conversion_cast %subview : memref<1x16x30x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x30x1280xf16>
            %subview_0 = memref.subview %arg1[0, 0, %arg6, %arg7] [1, 16, 28, 1280] [1, 1, 1, 1] : memref<1x16x?x?xf16> to memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>>
            %15 = builtin.unrealized_conversion_cast %subview_0 : memref<1x16x28x1280xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x16x28x1280xf16>
            %16 = Core.NestedCall @Module0::@main_func0_dims_H_cases_0_static(%14, %15) : (memref<1x16x30x1280xf16>, memref<1x16x28x1280xf16>) -> memref<1x16x28x1280xf16>
            scf.yield %16 : memref<1x16x28x1280xf16>
          }
        }
      }
      return %arg1 : memref<1x16x?x?xf16>

      // CHECK: module [[MODULE0:@.+]] {
      // CHECK: func.func private [[FUNC0:@.+]](%arg0: memref<1x16x30x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
      // CHECK: module [[MODULE1:@.+]] {
      // CHECK: func.func private [[FUNC1:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)
      // CHECK: module [[MODULE2:@.+]] {
      // CHECK: func.func private [[FUNC2:@.+]](%arg0: memref<1x16x29x1280xf16>, %arg1: memref<1x16x28x1280xf16>)

      // CHECK: func.func @AwaitAllCheckForLoop2D([[ARG0:%.+]]: memref<1x16x?x?xf16>, [[ARG1:%.+]]: memref<1x16x?x?xf16>, [[ARG2:%.+]]: index, [[ARG3:%.+]]: index, [[ARG4:%.+]]: index, [[ARG5:%.+]]: index)
      // CHECK: [[FALSE:%.+]] = arith.constant false
      // CHECK: [[C2:%.+]] = arith.constant 2 : index
      // CHECK: [[C3:%.+]] = arith.constant 3 : index
      // CHECK: [[C28:%.+]] = arith.constant 28 : index
      // CHECK: [[C1280:%.+]] = arith.constant 1280 : index
      // CHECK: [[C0:%.+]] = arith.constant 0 : index
      // CHECK: [[DIM_H:%.+]] = memref.dim [[ARG0]], [[C2]]
      // CHECK: [[DIM_W:%.+]] = memref.dim [[ARG0]], [[C3]]
      // CHECK: [[SUB:%.+]] = arith.subi [[DIM_H]], [[C0]]
      // CHECK: [[DIV:%.+]] = arith.divsi [[SUB]], [[C28]]
      // CHECK: [[GROUP:%.+]] = async.create_group [[DIV]]

      // CHECK: scf.for [[ARG6:%.+]] = [[C0]] to [[DIM_H]] step [[C28]] {
      // CHECK: scf.for [[ARG7:%.+]] = [[C0]] to [[DIM_W]] step [[C1280]] {

      // CHECK: [[CASE_H:%.+]] = arith.shli [[ARG2]], [[C2]] : index
      // CHECK: [[CASE:%.+]] = arith.ori [[CASE_H]], [[ARG4]] : index
      // CHECK: scf.index_switch [[CASE]]

      // CHECK: case 0 {
      // CHECK: async.add_to_group
      // CHECK-NOT: async.add_to_group
      // CHECK: async.await
      // CHECK-NOT: async.await

      // CHECK: case 1 {
      // CHECK: async.add_to_group
      // CHECK-NOT: async.add_to_group
      // CHECK: async.await
      // CHECK-NOT: async.await

      // CHECK: case 2 {
      // CHECK: async.add_to_group
      // CHECK-NOT: async.add_to_group
      // CHECK: async.await
      // CHECK-NOT: async.await

      // CHECK: default {
      // CHECK: cf.assert [[FALSE]], "Unsupported case"

      // CHECK: async.await_all [[GROUP]]
      // CHECK-NOT: }
      // CHECK-NOT: async.await_all
      // CHECK: return [[ARG1]]
    }
