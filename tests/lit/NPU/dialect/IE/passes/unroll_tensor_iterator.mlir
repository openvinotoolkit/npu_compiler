//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-tensor-iterator %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @UnrollTensorIterator
module @UnrollTensorIterator {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<2x3x4x5xf32>) -> (tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>) {
    %cst = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>
    %cst_0 = const.Declare tensor<2x3x4x5xf32> = dense<1.000000e+00> : tensor<2x3x4x5xf32>
    %0:2 = IE.TensorIterator body_module : {
    ^bb0(%arg2: tensor<1x4x6x10xf32>, %arg3: tensor<2x3x4x5xf32>):
      %1 = IE.Add(%arg3, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
      %2 = IE.Add(%arg2, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2, %1) : (tensor<1x4x6x10xf32>, tensor<2x3x4x5xf32>) -> ()
    } num_iterations : 3
    slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 2 : i64>]
    invariant_input_descs : []
    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, body_input_index = 1 : i64>]
    concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>]
    (%arg0, %arg1) : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32> -> tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>
    return %0#0, %0#1 : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x4x6x10xf32> = dense<1.000000e+00> : tensor<1x4x6x10xf32>

    // CHECK-DAG:       [[CST1:%.*]] = const.Declare tensor<2x3x4x5xf32> = dense<1.000000e+00> : tensor<2x3x4x5xf32>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [2, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

    // CHECK:       [[ADD0:%.*]] = IE.Add([[ARG1]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
    // CHECK:       [[ADDSLICE0:%.*]] = IE.Add([[SLICE0]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADD1:%.*]] = IE.Add([[ADD0]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
    // CHECK:       [[ADDSLICE1:%.*]] = IE.Add([[SLICE1]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADD2:%.*]] = IE.Add([[ADD1]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
    // CHECK:       [[ADDSLICE2:%.*]] = IE.Add([[SLICE2]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ADDSLICE0]], [[ADDSLICE1]], [[ADDSLICE2]])
    // CHECK-SAME:       {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:       : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<3x4x6x10xf32>

    // CHECK:       return [[CONCAT0]], [[ADD2]] : tensor<3x4x6x10xf32>, tensor<2x3x4x5xf32>

}

}

// -----

// CHECK-LABEL: @UnrollTensorIteratorNoConcat
module @UnrollTensorIteratorNoConcat {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>) {
    %cst_0 = const.Declare tensor<2x3x4x5xf32> = dense<1.000000e+00> : tensor<2x3x4x5xf32>
    %0 = IE.TensorIterator body_module : {
    ^bb0(%arg1: tensor<2x3x4x5xf32>):
      %1 = IE.Add(%arg1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
      "IE.LoopTerminator"(%1) : (tensor<2x3x4x5xf32>) -> ()
    }
    num_iterations : 3
    slice_input_descs : []
    invariant_input_descs : []
    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
    concat_output_descs : []
    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
    (%arg0) : tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
    return %0: tensor<2x3x4x5xf32>

    // CHECK-DAG:   [[CST1:%.*]] = const.Declare tensor<2x3x4x5xf32> = dense<1.000000e+00> : tensor<2x3x4x5xf32>

    // CHECK:       [[ADD0:%.*]] = IE.Add([[ARG0]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADD1:%.*]] = IE.Add([[ADD0]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADD2:%.*]] = IE.Add([[ADD1]], [[CST1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       return [[ADD2]] : tensor<2x3x4x5xf32>

}

}


// -----

// CHECK-LABEL: @UnrollTensorIteratorAllConcat
module @UnrollTensorIteratorAllConcat {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<2x3x4x5xf32>)
func.func @main(%arg0: tensor<2x3x4x5xf32>) -> (tensor<2x15x4x5xf32>) {
    %cst_0 = const.Declare tensor<2x3x4x5xf32> = dense<1.000000e+00> : tensor<2x3x4x5xf32>
    %0 = IE.TensorIterator body_module : {
    ^bb0(%arg1: tensor<2x3x4x5xf32>):
      %1 = IE.Add(%arg1, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>
      "IE.LoopTerminator"(%1) : (tensor<2x3x4x5xf32>) -> ()
    }
    num_iterations : 5
    slice_input_descs : []
    invariant_input_descs : []
    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, body_input_index = 0 : i64>]
    concat_output_descs : [#IE.ConcatOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 1 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = -1 : i64>]
    invariant_output_descs : []
    (%arg0) : tensor<2x3x4x5xf32> -> tensor<2x15x4x5xf32>
    return %0: tensor<2x15x4x5xf32>

    // CHECK-DAG:   [[CST0:%.*]] = const.Declare tensor<2x3x4x5xf32> = dense<1.000000e+00> : tensor<2x3x4x5xf32>

    // CHECK:       [[ADD0:%.*]] = IE.Add([[ARG0]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADD1:%.*]] = IE.Add([[ADD0]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADD2:%.*]] = IE.Add([[ADD1]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADD3:%.*]] = IE.Add([[ADD2]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[ADD4:%.*]] = IE.Add([[ADD3]], [[CST0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x3x4x5xf32>

    // CHECK:       [[CONCAT0:%.+]] = IE.Concat([[ADD0]], [[ADD1]], [[ADD2]], [[ADD3]], [[ADD4]])
    // CHECK-SAME:      {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      : tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32> -> tensor<2x15x4x5xf32>

    // CHECK:       return [[CONCAT0]] : tensor<2x15x4x5xf32>
}

}

// -----

// CHECK-LABEL: @UnrollTensorIterator3Input1Output
module @UnrollTensorIterator3Input1Output {

// CHECK:       func.func @main(
// CHECK-SAME:      [[ARG0:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG1:%arg[0-9]+]]: tensor<3x4x6x10xf32>,
// CHECK-SAME:      [[ARG2:%arg[0-9]+]]: tensor<1x4x6x10xf32>)
func.func @main(%arg0: tensor<3x4x6x10xf32>, %arg1: tensor<3x4x6x10xf32>, %arg2: tensor<1x4x6x10xf32>) -> (tensor<1x4x6x10xf32>) {
    %0 = IE.TensorIterator body_module : {
    ^bb0(%arg3: tensor<1x4x6x10xf32>, %arg4: tensor<1x4x6x10xf32>, %arg5: tensor<1x4x6x10xf32>):
      %1 = IE.Add(%arg3, %arg4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      %2 = IE.Add(%arg5, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
      "IE.LoopTerminator"(%2) : (tensor<1x4x6x10xf32>) -> ()
    }
    num_iterations : 3
    slice_input_descs : [#IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, axis = 0 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 2 : i64>, #IE.SliceInputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, axis = 0 : i64, start = 2 : i64, stride = -1 : i64, part_size = 1 : i64, end = 0 : i64>]
    invariant_input_descs : []
    feedback_input_descs : [#IE.MergedInputPortMap<external_port_id = 2 : i64, internal_layer_id = 2 : i64, body_input_index = 0 : i64>]
    concat_output_descs : []
    invariant_output_descs : [#IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>]
    (%arg0, %arg1, %arg2) : tensor<3x4x6x10xf32>, tensor<3x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
    return %0 : tensor<1x4x6x10xf32>

    // CHECK:       [[SLICE0:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE2:%.+]] = IE.Slice [[ARG0]]
    // CHECK-SAME:      [2, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

    // CHECK:       [[SLICE3:%.+]] = IE.Slice [[ARG1]]
    // CHECK-SAME:      [2, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE4:%.+]] = IE.Slice [[ARG1]]
    // CHECK-SAME:      [1, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>
    // CHECK:       [[SLICE5:%.+]] = IE.Slice [[ARG1]]
    // CHECK-SAME:      [0, 0, 0, 0] [1, 4, 6, 10] : tensor<3x4x6x10xf32> to tensor<1x4x6x10xf32>

    // CHECK:       [[ADDSLICE0:%.*]] = IE.Add([[SLICE0]], [[SLICE3]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
    // CHECK:       [[ADD0:%.*]] = IE.Add([[ARG2]], [[ADDSLICE0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADDSLICE1:%.*]] = IE.Add([[SLICE1]], [[SLICE4]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
    // CHECK:       [[ADD1:%.*]] = IE.Add([[ADD0]], [[ADDSLICE1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       [[ADDSLICE2:%.*]] = IE.Add([[SLICE2]], [[SLICE5]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>
    // CHECK:       [[ADD2:%.*]] = IE.Add([[ADD1]], [[ADDSLICE2]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      : tensor<1x4x6x10xf32>, tensor<1x4x6x10xf32> -> tensor<1x4x6x10xf32>

    // CHECK:       return [[ADD2]] : tensor<1x4x6x10xf32>

}

}
