//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-translate --vpu-arch=%arch% --import-IE ./loop.xml | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
// CHECK-SAME:      -> (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>) {

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<2> : tensor<1xsi32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>

// CHECK:           [[LOOP:%.*]]:2 = IE.Loop([[CST]], [[CST0]], [[ARG0]], [[ARG1]])
// CHECK:           : tensor<1xsi32>, tensor<1xi8>, tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
// CHECK:           (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
// CHECK:           slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
// CHECK:           concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG2:%arg[0-9]+]]: tensor<1x4x6x10xf32>, [[ARG3:%arg[0-9]+]]: tensor<2x3x4x5xf32>):

// CHECK-DAG:       [[CST1:%.*]] = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>
// CHECK-DAG:       [[CST2:%.*]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK:           [[ADD1:%.*]] = IE.Add([[ARG3]], [[CST2]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>

// CHECK-DAG:       [[CST3:%.*]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
// CHECK:           [[ADD2:%.*]] = IE.Add([[ARG2]], [[CST3]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
// CHECK:           "IE.LoopTerminator"([[ADD2]], [[ADD1]], [[CST1]]) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
// CHECK:           }

// CHECK:           return [[LOOP]]#0, [[LOOP]]#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>

// CHECK:   }
