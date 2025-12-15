//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-tensor-iterator %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollLoop_ConstTrue_InternalExecCond
module @UnrollLoop_ConstTrue_InternalExecCond {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>) -> (tensor<2x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst_2 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0:2 = IE.Loop(%arg0, %arg1) : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<2x4x6x10xf32>, tensor<2x3x4x5xf32>
     (num_iterations : 2 current_iter_index : -1 exec_cond_index : 2)
     slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 2 : i64>]
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
     concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
    body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>):
      %1 = IE.Add(%arg3, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      %2 = IE.Add(%arg2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1, %cst_2) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0#0, %0#1 : tensor<2x4x6x10xf32>, tensor<2x3x4x5xf32>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

    // CHECK:       [[ADD0:%.+]] = IE.Add([[ARG1]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
    // CHECK:       [[ADDSLICE0:%.+]] = IE.Add([[SLICE0]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADD1:%.+]] = IE.Add([[ADD0]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
    // CHECK:       [[ADDSLICE1:%.+]] = IE.Add([[SLICE1]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ADDSLICE0]], [[ADDSLICE1]])
    // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:            : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<2x4x6x10xf32>

    // CHECK:       return [[CONCAT0]], [[ADD1]] : tensor<2x4x6x10xf32>, tensor<2x3x4x5xf32>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollLoop_ConstFalse_InternalExecCond
module @UnrollLoop_ConstFalse_InternalExecCond {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>) -> (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    %cst_0 = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>
    %cst_1 = const.Declare tensor<1xi8> = dense<0> : tensor<1xi8>
    %0:2 = IE.Loop(%arg0, %arg1) : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
    (num_iterations : 1 current_iter_index : -1 exec_cond_index : 2)
     slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 3 : i64, end = 2 : i64>]
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
     concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>):
      %1 = IE.Add(%arg3, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
      %2 = IE.Add(%arg2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1, %cst_1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0#0, %0#1 : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<1.000000e+00> : tensor<1x1x1x1xf32>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

    // CHECK:       [[ADD0:%.+]] = IE.Add([[ARG1]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1x1x1x1xf32> -> tensor<2x3x4x5xf32>
    // CHECK:       [[ADDSLICE0:%.+]] = IE.Add([[SLICE0]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       return [[ADDSLICE0]], [[ADD0]] : tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollLoop_UseCurIterAsParam
module @UnrollLoop_UseCurIterAsParam {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>) -> (tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    %cst_1 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0:2 = IE.Loop(%arg0, %arg1) : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>
    (num_iterations : 3 current_iter_index : 2 exec_cond_index : 2)
     slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 2 : i64>]
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
     concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
    body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>, %arg4: tensor<1xf32>):
      %1 = IE.Add(%arg3, %arg4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
      %2 = IE.Add(%arg2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1, %cst_1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>) -> ()
    }
    return %0#0, %0#1 : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK-DAG:       [[CST2:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [2, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

    // CHECK:       [[ADDSLICE0:%.+]] = IE.Add([[SLICE0]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADD1:%.+]] = IE.Add([[ARG1]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADDSLICE1:%.+]] = IE.Add([[SLICE1]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADD2:%.+]] = IE.Add([[ADD1]], [[CST2]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADDSLICE2:%.+]] = IE.Add([[SLICE2]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ADDSLICE0]], [[ADDSLICE1]], [[ADDSLICE2]])
    // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:            : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<3x4x6x10xf32>

    // CHECK:       return [[CONCAT0]], [[ADD2]] : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>
}

}


// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollLoop_2MergedInput_2InvariantOutput
module @UnrollLoop_2MergedInput_2InvariantOutput {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>, %arg2: tensor<2x3x4x5xf32>) -> (tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) {
  %0:3 = IE.Loop(%arg0, %arg1, %arg2) : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>
  (num_iterations : 3 current_iter_index : 3 exec_cond_index : 2)
    slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 2 : i64>]
    invariant_input_descs : []
    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>, #IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 2 : i64, body_input_index = 3 : i64>]
    concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>, #IE.InvariantOutputPortMap<external_port_id = 2 : i64, internal_layer_id = 3 : i64, iterations = -1 : i64>]
    body_module : {
  ^bb0(%arg3: tensor<1x4x6x10xf32>, %arg4: tensor<2x3x4x5xf32>, %arg5: tensor<2x3x4x5xf32>, %arg6: tensor<1xf32>):
    %1 = IE.Add(%arg5, %arg6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
    %cst_1 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %2 = IE.Add(%arg4, %arg6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
    %cst_2 = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    %3 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
    "IE.LoopTerminator"(%3, %2, %cst_1, %1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>, tensor<2x3x4x5xf32>) -> ()
  }
  return %0#0, %0#1, %0#2 : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>

}

  // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
  // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>

  // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
  // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
  // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]]
  // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
  // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG0]]
  // CHECK-SAME:      [2, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

  // CHECK:       [[ADDSLICE0:%.+]] = IE.Add([[SLICE0]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
  // CHECK:       [[ADD4:%.+]] = IE.Add([[ARG2]], [[CST0]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
  // CHECK:       [[ADD5:%.+]] = IE.Add([[ARG1]], [[CST0]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>

  // CHECK:       [[ADDSLICE1:%.+]] = IE.Add([[SLICE1]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
  // CHECK:       [[ADD7:%.+]] = IE.Add([[ADD4]], [[CST]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
  // CHECK:       [[ADD8:%.+]] = IE.Add([[ADD5]], [[CST]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>

  // CHECK:       [[ADDSLICE2:%.+]] = IE.Add([[SLICE2]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

  // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ADDSLICE0]], [[ADDSLICE1]], [[ADDSLICE2]])
  // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK-SAME:            : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<3x4x6x10xf32>

  // CHECK:       return [[CONCAT0]], [[ADD8]], [[ADD7]] : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>

}

// -----
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollLoop_2InvariantInput_3ConcatOutput
module @UnrollLoop_2InvariantInput_3ConcatOutput {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>, %arg2: tensor<2x3x4x5xf32>) -> (tensor<3x4x6x10xf32>, tensor<6x3x4x5xf32>, tensor<6x3x4x5xf32>) {
  %0:3 = IE.Loop(%arg0, %arg1, %arg2) : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<3x4x6x10xf32>, tensor<6x3x4x5xf32>, tensor<6x3x4x5xf32>
  (num_iterations : 3 current_iter_index : 3 exec_cond_index : 2)
    slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 2 : i64>]
    invariant_input_descs : [#IE.InvariantInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64>, #IE.InvariantInputPortMap<external_port_id = 2 : i64, internal_layer_id = 2 : i64>]
    feedback_input_descs : []
    concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>, #IE.ConcatOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>, #IE.ConcatOutputPortMap<external_port_id = 2 : i64, internal_layer_id = 3 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
    invariant_output_descs : []
    body_module : {
  ^bb0(%arg3: tensor<1x4x6x10xf32>, %arg4: tensor<2x3x4x5xf32>, %arg5: tensor<2x3x4x5xf32>, %arg6: tensor<1xf32>):
    %1 = IE.Add(%arg5, %arg6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
    %cst_1 = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %2 = IE.Add(%arg4, %arg6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
    %cst_2 = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    %3 = IE.Add(%arg3, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
    "IE.LoopTerminator"(%3, %2, %cst_1, %1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>, tensor<1xi8>, tensor<2x3x4x5xf32>) -> ()
  }
  return %0#0, %0#1, %0#2 : tensor<3x4x6x10xf32>, tensor<6x3x4x5xf32>, tensor<6x3x4x5xf32>
}

  // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
  // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>

  // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
  // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
  // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]]
  // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
  // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG0]]
  // CHECK-SAME:      [2, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

  // CHECK:       [[ADDSLICE0:%.+]] = IE.Add([[SLICE0]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
  // CHECK:       [[ADD4:%.+]] = IE.Add([[ARG2]], [[CST0]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
  // CHECK:       [[ADD5:%.+]] = IE.Add([[ARG1]], [[CST0]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>

  // CHECK:       [[ADDSLICE1:%.+]] = IE.Add([[SLICE1]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
  // CHECK:       [[ADD7:%.+]] = IE.Add([[ARG2]], [[CST]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>
  // CHECK:       [[ADD8:%.+]] = IE.Add([[ARG1]], [[CST]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<1xf32> -> tensor<2x3x4x5xf32>

  // CHECK:       [[ADDSLICE2:%.+]] = IE.Add([[SLICE2]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
  // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

  // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ADDSLICE0]], [[ADDSLICE1]], [[ADDSLICE2]])
  // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK-SAME:            : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<3x4x6x10xf32>
  // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[ARG1]], [[ADD5]], [[ADD8]])
  // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK-SAME:            : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<6x3x4x5xf32>
  // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[ARG2]], [[ADD4]], [[ADD7]])
  // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK-SAME:            : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<6x3x4x5xf32>

  // CHECK:       return [[CONCAT0]], [[CONCAT1]], [[CONCAT2]] : tensor<3x4x6x10xf32>, tensor<6x3x4x5xf32>, tensor<6x3x4x5xf32>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollLoop_ParamInternalExecCond
module @UnrollLoop_ParamInternalExecCond {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<1x5x3xsi32>)
func.func @main(%arg0: tensor<1x5x3xsi32>) -> tensor<1x5x3xsi32> {
    %cst = const.Declare tensor<1x1x1xsi32> = dense<4> : tensor<1x1x1xsi32>
    %cst_0 = const.Declare tensor<1xsi32> = dense<4> : tensor<1xsi32>
    %cst_1 = const.Declare tensor<1xsi32> = dense<3> : tensor<1xsi32>
    %cst_2 = const.Declare tensor<1xsi8> = dense<1> : tensor<1xsi8>
    %0 = IE.Loop(%cst_1, %cst_2, %arg0) : tensor<1xsi32>, tensor<1xsi8>, tensor<1x5x3xsi32> -> tensor<1x5x3xsi32>
    (num_iterations : 3 current_iter_index : 1 exec_cond_index : 1)
     slice_input_descs : []
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
     concat_output_descs : []
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg1: tensor<1x5x3xsi32>, %arg2: tensor<1xsi32>):
      %1 = IE.Less(%arg2, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
      %2 = IE.Add(%arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>
      "IE.LoopTerminator"(%2, %1) : (tensor<1x5x3xsi32>, tensor<1xi8>) -> ()
    }
    return %0 : tensor<1x5x3xsi32>

  // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<2.000000e+00> : tensor<1xf32>, [#const.CastElemType<si32>]
  // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1xsi32> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<si32>]
  // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x1x1xsi32> = dense<4> : tensor<1x1x1xsi32>
  // CHECK-DAG:       [[CST2:%.+]] = const.Declare tensor<1xsi32> = dense<4> : tensor<1xsi32>
  // CHECK-DAG:       [[CST3:%.+]] = const.Declare tensor<1xsi8> = dense<1> : tensor<1xsi8>
  // CHECK-DAG:       [[CST4:%.+]] = const.Declare tensor<1xsi32> = dense<0.000000e+00> : tensor<1xf32>, [#const.CastElemType<si32>]

  // CHECK:           [[LESS0:%.+]] = IE.Less([[CST4]], [[CST2]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:      tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
  // CHECK:           [[ADD0:%.+]] = IE.Add([[ARG0]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:      tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>

  // CHECK:           [[LESS1:%.+]] = IE.Less([[CST0]], [[CST2]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:      tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
  // CHECK:           [[ADD1:%.+]] = IE.Add([[ADD0]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:      tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>

  // CHECK:           [[LESS2:%.+]] = IE.Less([[CST]], [[CST2]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:      tensor<1xsi32>, tensor<1xsi32> -> tensor<1xi8>
  // CHECK:           [[ADD2:%.+]] = IE.Add([[ADD1]], [[CST1]])
  // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
  // CHECK-SAME:      tensor<1x5x3xsi32>, tensor<1x1x1xsi32> -> tensor<1x5x3xsi32>

  // CHECK:           [[CONCAT0:%.+]] = IE.Concat([[LESS0]], [[LESS1]], [[LESS2]])
  // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK-SAME:            : tensor<1xi8>, tensor<1xi8>, tensor<1xi8> -> tensor<3xi8>

  // CHECK:           [[CONCAT1:%.+]] = IE.Concat([[ADD0]], [[ADD1]], [[ADD2]])
  // CHECK-SAME:            {per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK-SAME:            : tensor<1x5x3xsi32>, tensor<1x5x3xsi32>, tensor<1x5x3xsi32> -> tensor<3x5x3xsi32>

  // CHECK:           [[LOOPSELECT:%.+]] = IE.LoopSelect([[CST3]], [[CONCAT0]], [[CONCAT1]])
  // CHECK-SAME{LITERAL}:   {axis = 0 : i64, do_concat = false, stride = 1 : i64} :
  // CHECK-SAME:            tensor<1xsi8>, tensor<3xi8>, tensor<3x5x3xsi32> -> tensor<1x5x3xsi32>

  // CHECK:           return [[LOOPSELECT]] : tensor<1x5x3xsi32>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @UnrollLoopWithOutliner {
// CHECK-LABEL: @UnrollLoopWithOutliner

  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "Parameter_3" tensorNames = ["Parameter_3"] : tensor<1x1x2x3xf32>
  } outputsInfo : {
    DataInfo "Loop_17" friendlyName = "Result_18" : tensor<1x1x2x3xf32>
  }

  func.func private @main_loop_body1(%arg0: tensor<1x1x2x3xf32>) -> (tensor<1x1x2x3xf32>, tensor<1xi8>) {
    %cst = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0 = IE.SoftMax(%arg0) {axisInd = 2 : i64} : tensor<1x1x2x3xf32> -> tensor<1x1x2x3xf32>
    return %0, %cst : tensor<1x1x2x3xf32>, tensor<1xi8>
  }
// CHECK:  func.func private @main_loop_body1([[ARG0:%.+]]: tensor<1x1x2x3xf32>)
// CHECK-SAME:    -> (tensor<1x1x2x3xf32>, tensor<1xi8>) {
// CHECK:  [[CONST:%.+]] = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
// CHECK:  [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 2 : i64} : tensor<1x1x2x3xf32> -> tensor<1x1x2x3xf32>
// CHECK:  return [[SOFTMAX]], [[CONST]] : tensor<1x1x2x3xf32>, tensor<1xi8>
// CHECK:  }

  func.func @main(%arg0: tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32> {
    %cst = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
    %0 = IE.Loop(%arg0) : tensor<1x1x2x3xf32> -> tensor<1x1x2x3xf32>
    (num_iterations : 2 current_iter_index : -1 exec_cond_index : 1)
     slice_input_descs : []
     invariant_input_descs : []
     feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
     concat_output_descs : []
     invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
     body_module : {
    ^bb0(%arg1: tensor<1x1x2x3xf32>):
      %1:2 = func.call @main_loop_body1(%arg1) : (tensor<1x1x2x3xf32>) -> (tensor<1x1x2x3xf32>, tensor<1xi8>)
      "IE.LoopTerminator"(%1#0, %cst) : (tensor<1x1x2x3xf32>, tensor<1xi8>) -> ()
    }
    return %0 : tensor<1x1x2x3xf32>
  }
// CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32> {
// CHECK:  [[CALL0:%.+]]:2 = call @main_loop_body1([[ARG0]]) : (tensor<1x1x2x3xf32>) -> (tensor<1x1x2x3xf32>, tensor<1xi8>)
// CHECK:  [[CALL1:%.+]]:2 = call @main_loop_body1([[CALL0]]#0) : (tensor<1x1x2x3xf32>) -> (tensor<1x1x2x3xf32>, tensor<1xi8>)
// CHECK:  return [[CALL1]]#0 : tensor<1x1x2x3xf32>
// CHECK:  }

}
