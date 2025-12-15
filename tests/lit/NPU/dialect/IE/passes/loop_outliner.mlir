//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --loop-outliner %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


module @LoopOutlinerWith1Loop {
// CHECK-LABEL: @LoopOutlinerWith1Loop

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "Parameter_2" tensorNames = ["Parameter_2"] : tensor<1xsi64>
  DataInfo "Parameter_5" tensorNames = ["Parameter_5"] : tensor<3x5xf32>
  DataInfo "Parameter_6" tensorNames = ["Parameter_6"] : tensor<2x5xf32>
} outputsInfo : {
  DataInfo "Loop_19.0" friendlyName = "Result_20" : tensor<3x5xf32>
  DataInfo "Loop_19.1" friendlyName = "Result_21" : tensor<2x5xf32>
}

func.func @main(%arg0: tensor<1xsi64>, %arg1: tensor<3x5xf32>, %arg2: tensor<2x5xf32>) -> (tensor<3x5xf32>, tensor<2x5xf32>) {
  %cst_2 = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
  %cst_3 = const.Declare tensor<1xsi64> = dense<9> : tensor<1xsi64>
  %0 = IE.Less(%arg0, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
  %1:2 = IE.Loop(%cst_2, %0, %arg1, %arg2) : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32>, tensor<2x5xf32> -> tensor<3x5xf32>, tensor<2x5xf32>
  (num_iterations : 5 current_iter_index : 2 exec_cond_index : 2)
   slice_input_descs : []
   invariant_input_descs : []
   feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>, #IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
   concat_output_descs : []
   invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>, #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
   body_module : {
  ^bb0(%arg3: tensor<3x5xf32>, %arg4: tensor<2x5xf32>, %arg5: tensor<1xsi64>):
    %cst = const.Declare tensor<3x5xf32> = dense<1.000000e+00> : tensor<3x5xf32>
    %cst_0 = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
    %cst_1 = const.Declare tensor<1xsi64> = dense<4> : tensor<1xsi64>
    %2 = IE.Less(%arg5, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
    %3 = IE.Add(%arg4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x5xf32>, tensor<1x1xf32> -> tensor<2x5xf32>
    %4 = IE.Add(%arg3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>
    "IE.LoopTerminator"(%4, %3, %2) : (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>) -> ()
  }
  return %1#0, %1#1 : tensor<3x5xf32>, tensor<2x5xf32>
}

// CHECK:  func.func private @main_loop_body1([[ARG0:%.+]]: tensor<3x5xf32>, [[ARG1:%.+]]: tensor<2x5xf32>, [[ARG2:%.+]]: tensor<1xsi64>)
// CHECK-SAME:    -> (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>) {

// CHECK:  [[CONST:%.+]] = const.Declare tensor<3x5xf32> = dense<1.000000e+00> : tensor<3x5xf32>
// CHECK:  [[CONST0:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
// CHECK:  [[CONST1:%.+]] = const.Declare tensor<1xsi64> = dense<4> : tensor<1xsi64>

// CHECK:  [[LESS:%.+]] = IE.Less([[ARG2]], [[CONST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
// CHECK:  [[ADD1:%.+]] = IE.Add([[ARG1]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<2x5xf32>, tensor<1x1xf32> -> tensor<2x5xf32>
// CHECK:  [[ADD2:%.+]] = IE.Add([[ARG0]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<3x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>
// CHECK:  return [[ADD2]], [[ADD1]], [[LESS]] : tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>
// CHECK:  }

// CHECK:  func.func @main([[ARG0:%.+]]: tensor<1xsi64>, [[ARG1:%.+]]: tensor<3x5xf32>, [[ARG2:%.+]]: tensor<2x5xf32>)
// CHECK-SAME:    -> (tensor<3x5xf32>, tensor<2x5xf32>) {

// CHECK:  [[CONST:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
// CHECK:  [[CONST0:%.+]] = const.Declare tensor<1xsi64> = dense<9> : tensor<1xsi64>

// CHECK:  [[LESS:%.+]] = IE.Less([[ARG0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
// CHECK:  [[LOOP:%.+]]:2 = IE.Loop([[CONST]], [[LESS]], [[ARG1]], [[ARG2]])
// CHECK-SAME:    : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32>, tensor<2x5xf32> -> tensor<3x5xf32>, tensor<2x5xf32>
// CHECK:  (num_iterations : 5 current_iter_index : 2 exec_cond_index : 2)
// CHECK:  slice_input_descs : []
// CHECK:  invariant_input_descs : []
// CHECK:  feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>,
// CHECK-SAME:    #IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
// CHECK:  concat_output_descs : []
// CHECK:  invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>,
// CHECK-SAME:    #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]

// CHECK:  body_module : {
// CHECK:  ^bb0([[ARG3:%.+]]: tensor<3x5xf32>, [[ARG4:%.+]]: tensor<2x5xf32>, [[ARG5:%.+]]: tensor<1xsi64>):
// CHECK:     [[CALL:%.+]]:3 = func.call @main_loop_body1([[ARG3]], [[ARG4]], [[ARG5]])
// CHECK-SAME:    (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xsi64>) -> (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>)
// CHECK:     "IE.LoopTerminator"([[CALL]]#0, [[CALL]]#1, [[CALL]]#2) : (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>) -> ()
// CHECK:   }
// CHECK:   return [[LOOP]]#0, [[LOOP]]#1 : tensor<3x5xf32>, tensor<2x5xf32>
// CHECK:  }

}

// -----

module @LoopOutlinerWith2Loop {
// CHECK-LABEL: @LoopOutlinerWith2Loop

net.NetworkInfo entryPoint : @main inputsInfo : {
  DataInfo "Parameter_2" tensorNames = ["Parameter_2"] : tensor<1xsi64>
  DataInfo "Parameter_5" tensorNames = ["Parameter_5"] : tensor<3x5xf32>
  DataInfo "Parameter_6" tensorNames = ["Parameter_6"] : tensor<2x5xf32>
} outputsInfo : {
  DataInfo "Loop_19.0" friendlyName = "Result_20" : tensor<3x5xf32>
}

func.func @main(%arg0: tensor<1xsi64>, %arg1: tensor<3x5xf32>, %arg2: tensor<2x5xf32>) -> (tensor<3x5xf32>) {
  %cst_2 = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
  %cst_3 = const.Declare tensor<1xsi64> = dense<9> : tensor<1xsi64>
  %0 = IE.Less(%arg0, %cst_3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
  %1:2 = IE.Loop(%cst_2, %0, %arg1, %arg2) : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32>, tensor<2x5xf32> -> tensor<3x5xf32>, tensor<2x5xf32>
  (num_iterations : 5 current_iter_index : 2 exec_cond_index : 2)
   slice_input_descs : []
   invariant_input_descs : []
   feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>, #IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
   concat_output_descs : []
   invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>, #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
   body_module : {
  ^bb0(%arg3: tensor<3x5xf32>, %arg4: tensor<2x5xf32>, %arg5: tensor<1xsi64>):
    %cst = const.Declare tensor<3x5xf32> = dense<1.000000e+00> : tensor<3x5xf32>
    %cst_0 = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
    %cst_1 = const.Declare tensor<1xsi64> = dense<4> : tensor<1xsi64>
    %2 = IE.Less(%arg5, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
    %3 = IE.Add(%arg4, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x5xf32>, tensor<1x1xf32> -> tensor<2x5xf32>
    %4 = IE.Add(%arg3, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>
    "IE.LoopTerminator"(%4, %3, %2) : (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>) -> ()
  }
  %cst_5 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
  %cst_6 = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
  %5 = IE.Loop(%cst_6, %cst_5, %1#0) : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32> -> tensor<3x5xf32>
  (num_iterations : 3 current_iter_index : -1 exec_cond_index : 1)
    slice_input_descs : []
    invariant_input_descs : []
    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
    concat_output_descs : []
    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
    body_module : {
  ^bb0(%arg6: tensor<3x5xf32>):
    %cst_4 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %6 = IE.SoftMax(%arg6) {axisInd = 0 : i64} : tensor<3x5xf32> -> tensor<3x5xf32>
    "IE.LoopTerminator"(%6, %cst_4) : (tensor<3x5xf32>, tensor<1xi8>) -> ()
  }
  return %5#0 : tensor<3x5xf32>
}

// CHECK:  func.func private @main_loop_body1([[ARG0:%.+]]: tensor<3x5xf32>, [[ARG1:%.+]]: tensor<2x5xf32>, [[ARG2:%.+]]: tensor<1xsi64>)
// CHECK-SAME:    -> (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>) {
// CHECK:  [[CONST:%.+]] = const.Declare tensor<3x5xf32> = dense<1.000000e+00> : tensor<3x5xf32>
// CHECK:  [[CONST0:%.+]] = const.Declare tensor<1x1xf32> = dense<1.000000e+00> : tensor<1x1xf32>
// CHECK:  [[CONST1:%.+]] = const.Declare tensor<1xsi64> = dense<4> : tensor<1xsi64>
// CHECK:  [[LESS:%.+]] = IE.Less([[ARG2]], [[CONST1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>
// CHECK:  [[ADD1:%.+]] = IE.Add([[ARG1]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<2x5xf32>, tensor<1x1xf32> -> tensor<2x5xf32>
// CHECK:  [[ADD2:%.+]] = IE.Add([[ARG0]], [[CONST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<3x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>
// CHECK:  return [[ADD2]], [[ADD1]], [[LESS]] : tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>
// CHECK:  }

// CHECK:  func.func private @main_loop_body2([[ARG0:%.+]]: tensor<3x5xf32>) -> tensor<3x5xf32> {
// CHECK:  [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 0 : i64} : tensor<3x5xf32> -> tensor<3x5xf32>
// CHECK:  return [[SOFTMAX]] : tensor<3x5xf32>
// CHECK:  }

// CHECK:  func.func @main
// CHECK-SAME:    ([[ARG0:%.+]]: tensor<1xsi64>, [[ARG1:%.+]]: tensor<3x5xf32>, [[ARG2:%.+]]: tensor<2x5xf32>)
// CHECK-SAME:    -> tensor<3x5xf32> {

// CHECK:  [[CONST:%.+]] = const.Declare tensor<1xsi32> = dense<5> : tensor<1xsi32>
// CHECK:  [[CONST0:%.+]] = const.Declare tensor<1xsi64> = dense<9> : tensor<1xsi64>
// CHECK:  [[LESS:%.+]] = IE.Less([[ARG0]], [[CONST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
// CHECK-SAME:    : tensor<1xsi64>, tensor<1xsi64> -> tensor<1xi8>

// CHECK:  [[LOOP0:%.+]]:2 = IE.Loop([[CONST]], [[LESS]], [[ARG1]], [[ARG2]])
// CHECK-SAME:    : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32>, tensor<2x5xf32> -> tensor<3x5xf32>, tensor<2x5xf32>

// CHECK:  (num_iterations : 5 current_iter_index : 2 exec_cond_index : 2)
// CHECK:  slice_input_descs : []
// CHECK:  invariant_input_descs : []
// CHECK:  feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>,
// CHECK-SAME:    #IE.MergedInputPortMap<external_port_id = 3 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
// CHECK:  concat_output_descs : []
// CHECK:  invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>,
// CHECK-SAME:    #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]

// CHECK:  body_module : {
// CHECK:  ^bb0([[ARG3:%.+]]: tensor<3x5xf32>, [[ARG4:%.+]]: tensor<2x5xf32>, [[ARG5:%.+]]: tensor<1xsi64>):
// CHECK:     [[CALL:%.+]]:3 = func.call @main_loop_body1([[ARG3]], [[ARG4]], [[ARG5]])
// CHECK-SAME:    (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xsi64>) -> (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>)
// CHECK:     "IE.LoopTerminator"([[CALL]]#0, [[CALL]]#1, [[CALL]]#2) : (tensor<3x5xf32>, tensor<2x5xf32>, tensor<1xi8>) -> ()
// CHECK:   }

// CHECK:  [[CONST1:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
// CHECK:  [[CONST2:%.+]] = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>

// CHECK:  [[LOOP1:%.+]] = IE.Loop([[CONST2]], [[CONST1]], [[LOOP0]]#0)
// CHECK-SAME:    : tensor<1xsi32>, tensor<1xi8>, tensor<3x5xf32> -> tensor<3x5xf32>

// CHECK:  (num_iterations : 3 current_iter_index : -1 exec_cond_index : 1)
// CHECK:  slice_input_descs : []
// CHECK:  invariant_input_descs : []
// CHECK:  feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
// CHECK:  concat_output_descs : []
// CHECK:  invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]

// CHECK:  body_module : {
// CHECK:  ^bb0([[ARG6:%.+]]: tensor<3x5xf32>):
// CHECK:  [[CALL1:%.+]] = func.call @main_loop_body2([[ARG6]])
// CHECK-SAME:    (tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK:  [[CONST3:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
// CHECK:     "IE.LoopTerminator"([[CALL1]], [[CONST3]]) : (tensor<3x5xf32>, tensor<1xi8>) -> ()
// CHECK:  }
// CHECK:  return [[LOOP1]] : tensor<3x5xf32>
// CHECK:  }
}
