//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --canonicalize --split-input-file --init-compiler="vpu-arch=%arch%" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
// -----

// CHECK-LABEL: @RemoveConstInLoopInput
module @RemoveConstInLoopInput {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>) -> (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1xsi32> = dense<2> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0:2 = IE.Loop(%cst, %cst_0, %arg0, %arg1) : tensor<1xsi32>, tensor<1xi8>, tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
    (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
     slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
     concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>):
      %cst_1 = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
      %1 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      %cst_3 = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
      %2 = IE.Add(%arg2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1, %cst_1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0#0, %0#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
  }

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>

// CHECK:           [[LOOP:%.+]]:2 = IE.Loop([[ARG0]], [[ARG1]])
// CHECK:           : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
// CHECK:           (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
// CHECK:           slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
// CHECK:           concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG2:%arg[0-9]+]]: tensor<1x4x6x10xf32>, [[ARG3:%arg[0-9]+]]: tensor<2x3x4x5xf32>):
// CHECK:           [[ADD1:%.+]] = IE.Add([[ARG3]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
// CHECK:           [[ADD2:%.+]] = IE.Add([[ARG2]], [[CST]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
// CHECK:           "IE.LoopTerminator"([[ADD2]], [[ADD1]], [[CST1]]) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]]#0, [[LOOP]]#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>

}

// -----

// CHECK-LABEL: @RemoveConstInLoopInput2
module @RemoveConstInLoopInput2 {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>) -> (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1xsi32> = dense<2> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0:2 = IE.Loop(%cst, %cst_0, %arg0, %arg1) : tensor<1xsi32>, tensor<1xi8>, tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
    (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
     slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
     concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>):
      %cst_1 = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
      %1 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      %cst_3 = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
      %2 = IE.Add(%arg2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1, %cst_1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0#0, %0#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
  }

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>

// CHECK:           [[LOOP:%.+]]:2 = IE.Loop([[ARG0]], [[ARG1]])
// CHECK:           : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
// CHECK:           (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
// CHECK:           slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
// CHECK:           concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG2:%arg[0-9]+]]: tensor<1x4x6x10xf32>, [[ARG3:%arg[0-9]+]]: tensor<2x3x4x5xf32>):
// CHECK:           [[ADD1:%.+]] = IE.Add([[ARG3]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
// CHECK:           [[ADD2:%.+]] = IE.Add([[ARG2]], [[CST]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
// CHECK:           "IE.LoopTerminator"([[ADD2]], [[ADD1]], [[CST1]]) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]]#0, [[LOOP]]#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>

}

// -----

// CHECK-LABEL: @DontRemoveConstInLoopInput
module @DontRemoveConstInLoopInput {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>,
// CHECK-SAME:      [[ARG4:%arg[0-9]+]]: tensor<1xsi32>,
// CHECK-SAME:      [[ARG5:%arg[0-9]+]]: tensor<1xi8>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>, %arg4: tensor<1xsi32>, %arg5: tensor<1xi8>) -> (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %0:2 = IE.Loop(%arg4, %arg5, %arg0, %arg1) : tensor<1xsi32>, tensor<1xi8>, tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
    (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
     slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
     concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>):
      %cst_1 = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
      %1 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      %cst_3 = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
      %2 = IE.Add(%arg2, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1, %cst_1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0#0, %0#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
  }

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>

// CHECK:           [[LOOP:%.+]]:2 = IE.Loop([[ARG4]], [[ARG5]], [[ARG0]], [[ARG1]])
// CHECK:           : tensor<1xsi32>, tensor<1xi8>, tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
// CHECK:           (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
// CHECK:           slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
// CHECK:           concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG2:%arg[0-9]+]]: tensor<1x4x6x10xf32>, [[ARG3:%arg[0-9]+]]: tensor<2x3x4x5xf32>):
// CHECK:           [[ADD1:%.+]] = IE.Add([[ARG3]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
// CHECK:           [[ADD2:%.+]] = IE.Add([[ARG2]], [[CST]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
// CHECK:           "IE.LoopTerminator"([[ADD2]], [[ADD1]], [[CST1]]) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]]#0, [[LOOP]]#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>

}

// -----

// CHECK-LABEL: @ChangeNumIterations1
module @ChangeNumIterations1 {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg1: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1xsi32> = dense<10> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0 = IE.Loop(%cst, %cst_0, %arg1) : tensor<1xsi32>, tensor<1xi8>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
    (num_iterations : 1 current_iter_index : -1 exec_cond_index : 1)
     slice_input_descs : []
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
     concat_output_descs : []
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg3: tensor<2x3x4x5xf32>):
      %cst_1 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
      %1 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      "IE.LoopTerminator"(%1, %cst_1) : (tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0 : tensor<2x3x4x5xf32>
  }

// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>

// CHECK:           [[LOOP:%.+]] = IE.Loop([[ARG1]])
// CHECK:           : tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
// CHECK:           (num_iterations : 10 current_iter_index : -1 exec_cond_index : 1)
// CHECK:           slice_input_descs : []
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
// CHECK:           concat_output_descs : []
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG3:%arg[0-9]+]]: tensor<2x3x4x5xf32>):
// CHECK:           [[ADD1:%.+]] = IE.Add([[ARG3]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
// CHECK:           "IE.LoopTerminator"([[ADD1]], [[CST1]]) : (tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]] : tensor<2x3x4x5xf32>

}

// -----

// CHECK-LABEL: @ChangeNumIterations2
module @ChangeNumIterations2 {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg1: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1xsi32> = dense<10> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0 = IE.Loop(%cst, %cst_0, %arg1) : tensor<1xsi32>, tensor<1xi8>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
    (num_iterations : 3 current_iter_index : -1 exec_cond_index : 1)
     slice_input_descs : []
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
     concat_output_descs : []
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg3: tensor<2x3x4x5xf32>):
      %cst_1 = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
      %1 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      "IE.LoopTerminator"(%1, %cst_1) : (tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0 : tensor<2x3x4x5xf32>
  }

// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>

// CHECK:           [[LOOP:%.+]] = IE.Loop([[ARG1]])
// CHECK:           : tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
// CHECK:           (num_iterations : 1 current_iter_index : -1 exec_cond_index : 1)
// CHECK:           slice_input_descs : []
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
// CHECK:           concat_output_descs : []
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG3:%arg[0-9]+]]: tensor<2x3x4x5xf32>):
// CHECK:           [[ADD1:%.+]] = IE.Add([[ARG3]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
// CHECK:           "IE.LoopTerminator"([[ADD1]], [[CST1]]) : (tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]] : tensor<2x3x4x5xf32>

}

// -----

// CHECK-LABEL: @ChangeNumIterations3_INFER
module @ChangeNumIterations3_INFER {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x5x3xsi32>)
func.func @main(%arg0: tensor<1x5x3xsi32>) -> tensor<1x5x3xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<7> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xsi8> = dense<1> : tensor<1xsi8>
    %0 = IE.Loop(%cst, %cst_0, %arg0) : tensor<1xsi32>, tensor<1xsi8>, tensor<1x5x3xsi32> -> tensor<1x5x3xsi32>
    (num_iterations : -1 current_iter_index : 1 exec_cond_index : 1)
     slice_input_descs : []
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
     concat_output_descs : []
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg1: tensor<1x5x3xsi32>, %arg2: tensor<1xsi32>):
      %cst_1 = const.Declare tensor<1xsi32> = dense<4> : tensor<1xsi32>
      %1 = IE.Less(%arg2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1xsi32> = dense<4> : tensor<1x1x1xsi32>
      %2 = IE.Add(%arg1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>
      "IE.LoopTerminator"(%2, %1) : (tensor<1x5x3xsi32>, tensor<1xi8>) -> ()
    }
    return %0 : tensor<1x5x3xsi32>
  }

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x1x1xsi32> = dense<4> : tensor<1x1x1xsi32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<1xsi32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xsi32> = dense<7> : tensor<1xsi32>
// CHECK-DAG:       [[CST2:%.+]] = const.Declare tensor<1xsi8> = dense<1> : tensor<1xsi8>

// CHECK:           [[LOOP:%.+]] = IE.Loop([[CST1]], [[CST2]], [[ARG0]])
// CHECK:           : tensor<1xsi32>, tensor<1xsi8>, tensor<1x5x3xsi32> -> tensor<1x5x3xsi32>
// CHECK:           (num_iterations : 7 current_iter_index : 1 exec_cond_index : 1)
// CHECK:           slice_input_descs : []
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
// CHECK:           concat_output_descs : []
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG1:%arg[0-9]+]]: tensor<1x5x3xsi32>, [[ARG2:%arg[0-9]+]]: tensor<1xsi32>):
// CHECK:           [[LESS:%.+]] = IE.Less([[ARG2]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
// CHECK:           [[ADD:%.+]] = IE.Add([[ARG1]], [[CST]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>
// CHECK:           "IE.LoopTerminator"([[ADD]], [[LESS]]) : (tensor<1x5x3xsi32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]] : tensor<1x5x3xsi32>

}

// -----

// CHECK-LABEL: @ChangeNumIterations3_SetUpperBound
module @ChangeNumIterations3_SetUpperBound {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x5x3xsi32>)
func.func @main(%arg0: tensor<1x5x3xsi32>) -> tensor<1x5x3xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<-1> : tensor<1xsi32>
    %cst_0 = const.Declare tensor<1xsi8> = dense<1> : tensor<1xsi8>
    %0 = IE.Loop(%cst, %cst_0, %arg0) : tensor<1xsi32>, tensor<1xsi8>, tensor<1x5x3xsi32> -> tensor<1x5x3xsi32>
    (num_iterations : -1 current_iter_index : 1 exec_cond_index : 1)
     slice_input_descs : []
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
     concat_output_descs : []
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg1: tensor<1x5x3xsi32>, %arg2: tensor<1xsi32>):
      %cst_1 = const.Declare tensor<1xsi32> = dense<4> : tensor<1xsi32>
      %1 = IE.Less(%arg2, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
      %cst_2 = const.Declare tensor<1x1x1xsi32> = dense<4> : tensor<1x1x1xsi32>
      %2 = IE.Add(%arg1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>
      "IE.LoopTerminator"(%2, %1) : (tensor<1x5x3xsi32>, tensor<1xi8>) -> ()
    }
    return %0 : tensor<1x5x3xsi32>
  }

// CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x1x1xsi32> = dense<4> : tensor<1x1x1xsi32>
// CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<1xsi32>
// CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xsi32> = dense<-1> : tensor<1xsi32>
// CHECK-DAG:       [[CST2:%.+]] = const.Declare tensor<1xsi8> = dense<1> : tensor<1xsi8>

// CHECK:           [[LOOP:%.+]] = IE.Loop([[CST1]], [[CST2]], [[ARG0]])
// CHECK:           : tensor<1xsi32>, tensor<1xsi8>, tensor<1x5x3xsi32> -> tensor<1x5x3xsi32>
// CHECK:           (num_iterations : 10 current_iter_index : 1 exec_cond_index : 1)
// CHECK:           slice_input_descs : []
// CHECK:           invariant_input_descs : []
// CHECK:           feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
// CHECK:           concat_output_descs : []
// CHECK:           invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]

// CHECK:           body_module : {
// CHECK:           ^bb0([[ARG1:%arg[0-9]+]]: tensor<1x5x3xsi32>, [[ARG2:%arg[0-9]+]]: tensor<1xsi32>):
// CHECK:           [[LESS:%.+]] = IE.Less([[ARG2]], [[CST0]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
// CHECK:           [[ADD:%.+]] = IE.Add([[ARG1]], [[CST]])
// CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
// CHECK-SAME:      tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>
// CHECK:           "IE.LoopTerminator"([[ADD]], [[LESS]]) : (tensor<1x5x3xsi32>, tensor<1xi8>) -> ()

// CHECK:           return [[LOOP]] : tensor<1x5x3xsi32>

}
