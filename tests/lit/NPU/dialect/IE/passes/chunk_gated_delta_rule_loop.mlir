//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --chunk-gated-delta-rule-loop="chunk-size=4" %s | FileCheck %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --chunk-gated-delta-rule-loop="chunk-size=2" --unroll-tensor-iterator %s | FileCheck %s --check-prefix=UNROLLED
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --chunk-gated-delta-rule-loop="chunk-size=4" %s | FileCheck %s --check-prefix=NEGATED
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// CHECK-LABEL: module @ChunkGDR_BetaOrientation
module @ChunkGDR_BetaOrientation {

// CHECK: func.func @main(
func.func @main(
    %k: tensor<1x1x4x2xf32>,
    %q: tensor<1x1x4x2xf32>,
    %v: tensor<1x1x4x2xf32>,
    %gate: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>,
    %state: tensor<1x1x2x2xf32>,
    %accum: tensor<1x1x4x2xf32>) -> (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>) {

  %0:2 = IE.Loop(%k, %q, %v, %gate, %beta, %state, %accum)
      : tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4xf32>, tensor<1x1x4xf32>, tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
      -> tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
      (num_iterations : 4 current_iter_index : 0 exec_cond_index : 2)
      slice_input_descs : [
          #IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 1 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 1 : i64, internal_layer_id = 2 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 3 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 3 : i64, internal_layer_id = 4 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 4 : i64, internal_layer_id = 5 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>
      ]
      invariant_input_descs : []
      feedback_input_descs : [
          #IE.MergedInputPortMap<external_port_id = 5 : i64, internal_layer_id = 6 : i64, body_input_index = 0 : i64>,
          #IE.MergedInputPortMap<external_port_id = 6 : i64, internal_layer_id = 7 : i64, body_input_index = 1 : i64>
      ]
      concat_output_descs : []
      invariant_output_descs : [
          #IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>,
          #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>
      ]
      body_module : {
            ^bb0(%iter: tensor<si32>,
           %k_t: tensor<1x1x1x2xf32>,
           %q_t: tensor<1x1x1x2xf32>,
           %v_t: tensor<1x1x1x2xf32>,
           %gate_t: tensor<1x1x1xf32>,
           %beta_t: tensor<1x1x1xf32>,
           %h_prev: tensor<1x1x2x2xf32>,
           %acc_prev: tensor<1x1x4x2xf32>):

        %gate4 = IE.Reshape(%gate_t) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
        %decay = IE.Exp(%gate4) : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
        %h_decay = IE.Multiply(%h_prev, %decay) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>

        %k_exp = IE.Reshape(%k_t) {shape_value = [1, 1, 2, 1]} : tensor<1x1x1x2xf32> -> tensor<1x1x2x1xf32>
        %hk_mul = IE.Multiply(%h_prev, %k_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x2xf32>
        %hk = IE.ReduceSum(%hk_mul) {axes_value = [2], keep_dims} : tensor<1x1x2x2xf32> -> tensor<1x1x1x2xf32>

        %delta = IE.Subtract(%v_t, %hk) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf32>
        %beta4 = IE.Reshape(%beta_t) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
        %scaled = IE.Multiply(%delta, %beta4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x2xf32>

        %scaled_exp = IE.Reshape(%scaled) {shape_value = [1, 1, 1, 2]} : tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf32>
        %upd = IE.Multiply(%k_exp, %scaled_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x1xf32>, tensor<1x1x1x2xf32> -> tensor<1x1x2x2xf32>
        %h_new = IE.Add(%h_decay, %upd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

        %q_exp = IE.Reshape(%q_t) {shape_value = [1, 1, 2, 1]} : tensor<1x1x1x2xf32> -> tensor<1x1x2x1xf32>
        %y_mul = IE.Multiply(%h_new, %q_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x2xf32>
        %y = IE.ReduceSum(%y_mul) {axes_value = [2], keep_dims} : tensor<1x1x2x2xf32> -> tensor<1x1x1x2xf32>

        %iter_index = IE.Reshape(%iter) {shape_value = [1]} : tensor<si32> -> tensor<1xsi32>
        %new_acc = IE.ScatterUpdate(%acc_prev, %iter_index, %y) {axis_value = 2 : i64} : tensor<1x1x4x2xf32>, tensor<1xsi32>, tensor<1x1x1x2xf32> -> tensor<1x1x4x2xf32>
        %cond = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
        "IE.LoopTerminator"(%h_new, %new_acc, %cond) : (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>, tensor<1xi8>) -> ()
      }

  return %0#0, %0#1 : tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
}

// Verify the new Loop uses concat_output_descs instead of ScatterUpdate
// CHECK:      [[NEWLOOP:%.+]]:2 = IE.Loop
// CHECK:      (num_iterations : 1 current_iter_index : 0 exec_cond_index : 0)

// Verify concat_output_descs is populated (not empty)
// CHECK:      concat_output_descs : [#IE.ConcatOutputPortMap<

// Verify body has 7 args (no output_accum feedback)
// CHECK:      ^bb0({{%.+}}: tensor<1xsi32>, {{%.+}}: tensor<1x1x1x4x2xf32>, {{%.+}}: tensor<1x1x1x4x2xf32>, {{%.+}}: tensor<1x1x1x4x2xf32>, {{%.+}}: tensor<1x1x1x4xf32>, {{%.+}}: tensor<1x1x1x4xf32>, {{%.+}}: tensor<1x1x2x2xf32>):

// Verify intra-chunk computation: both inclusive and exclusive CumSum for correct decay
// CHECK:      [[KCHUNK:%.+]] = IE.Reshape{{.*}} : tensor<1x1x1x4x2xf32> -> tensor<1x1x4x2xf32>
// CHECK:      IE.CumSum({{%.+}}) {axis_value = 2 : i64} : tensor<1x1x4xf32>
// CHECK:      IE.CumSum({{%.+}}) {axis_value = 2 : i64, exclusive} : tensor<1x1x4xf32>
// CHECK:      [[KBETA:%.+]] = IE.Multiply([[KCHUNK]], {{%[0-9]+}}) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x4x2xf32>, tensor<1x1x4x1xf32> -> tensor<1x1x4x2xf32>
// CHECK:      [[KKT:%.+]] = IE.MatMul([[KCHUNK]], [[KBETA]]) {transpose_b} : tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32> -> tensor<1x1x4x4xf32>
// CHECK-NOT:  IE.MatMul([[KBETA]], [[KCHUNK]]) {transpose_b}

// Verify no ScatterUpdate in the body
// CHECK-NOT:  IE.ScatterUpdate

// Verify output is reshaped from [B,H,1,L,dV]
// CHECK:      IE.Reshape{{.*}} -> tensor<1x1x1x4x2xf32>
// CHECK:      "IE.LoopTerminator"

}

// -----

// This test verifies the end-to-end effect of ChunkGatedDeltaRuleLoop + UnrollTensorIterator:
//   Original Loop with ScatterUpdate → chunked Loop with concat_output_descs → fully unrolled with IE.Concat

// UNROLLED-LABEL: module @ChunkGDR_Unrolled
module @ChunkGDR_Unrolled {

// UNROLLED: func.func @main(
func.func @main(
    %k: tensor<1x1x4x2xf32>,
    %q: tensor<1x1x4x2xf32>,
    %v: tensor<1x1x4x2xf32>,
    %gate: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>,
    %state: tensor<1x1x2x2xf32>,
    %accum: tensor<1x1x4x2xf32>) -> (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>) {

  %0:2 = IE.Loop(%k, %q, %v, %gate, %beta, %state, %accum)
      : tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4xf32>, tensor<1x1x4xf32>, tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
      -> tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
      (num_iterations : 4 current_iter_index : 0 exec_cond_index : 2)
      slice_input_descs : [
          #IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 1 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 1 : i64, internal_layer_id = 2 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 3 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 3 : i64, internal_layer_id = 4 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 4 : i64, internal_layer_id = 5 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>
      ]
      invariant_input_descs : []
      feedback_input_descs : [
          #IE.MergedInputPortMap<external_port_id = 5 : i64, internal_layer_id = 6 : i64, body_input_index = 0 : i64>,
          #IE.MergedInputPortMap<external_port_id = 6 : i64, internal_layer_id = 7 : i64, body_input_index = 1 : i64>
      ]
      concat_output_descs : []
      invariant_output_descs : [
          #IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>,
          #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>
      ]
      body_module : {
            ^bb0(%iter: tensor<si32>,
           %k_t: tensor<1x1x1x2xf32>,
           %q_t: tensor<1x1x1x2xf32>,
           %v_t: tensor<1x1x1x2xf32>,
           %gate_t: tensor<1x1x1xf32>,
           %beta_t: tensor<1x1x1xf32>,
           %h_prev: tensor<1x1x2x2xf32>,
           %acc_prev: tensor<1x1x4x2xf32>):

        %gate4 = IE.Reshape(%gate_t) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
        %decay = IE.Exp(%gate4) : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
        %h_decay = IE.Multiply(%h_prev, %decay) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>

        %k_exp = IE.Reshape(%k_t) {shape_value = [1, 1, 2, 1]} : tensor<1x1x1x2xf32> -> tensor<1x1x2x1xf32>
        %hk_mul = IE.Multiply(%h_prev, %k_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x2xf32>
        %hk = IE.ReduceSum(%hk_mul) {axes_value = [2], keep_dims} : tensor<1x1x2x2xf32> -> tensor<1x1x1x2xf32>

        %delta = IE.Subtract(%v_t, %hk) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf32>
        %beta4 = IE.Reshape(%beta_t) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
        %scaled = IE.Multiply(%delta, %beta4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x2xf32>

        %scaled_exp = IE.Reshape(%scaled) {shape_value = [1, 1, 1, 2]} : tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf32>
        %upd = IE.Multiply(%k_exp, %scaled_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x1xf32>, tensor<1x1x1x2xf32> -> tensor<1x1x2x2xf32>
        %h_new = IE.Add(%h_decay, %upd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

        %q_exp = IE.Reshape(%q_t) {shape_value = [1, 1, 2, 1]} : tensor<1x1x1x2xf32> -> tensor<1x1x2x1xf32>
        %y_mul = IE.Multiply(%h_new, %q_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x2xf32>
        %y = IE.ReduceSum(%y_mul) {axes_value = [2], keep_dims} : tensor<1x1x2x2xf32> -> tensor<1x1x1x2xf32>

        %iter_index = IE.Reshape(%iter) {shape_value = [1]} : tensor<si32> -> tensor<1xsi32>
        %new_acc = IE.ScatterUpdate(%acc_prev, %iter_index, %y) {axis_value = 2 : i64} : tensor<1x1x4x2xf32>, tensor<1xsi32>, tensor<1x1x1x2xf32> -> tensor<1x1x4x2xf32>
        %cond = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
        "IE.LoopTerminator"(%h_new, %new_acc, %cond) : (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>, tensor<1xi8>) -> ()
      }

  return %0#0, %0#1 : tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
}

// After chunk + unroll: Loop is fully unrolled, no ScatterUpdate remains, Concat replaces it

// Verify no Loop or ScatterUpdate in the output
// UNROLLED-NOT: IE.Loop
// UNROLLED-NOT: IE.ScatterUpdate

// Verify intra-chunk computation: CumSum for cumulative gate decay
// UNROLLED:     IE.CumSum
// UNROLLED:     IE.MatMul

// Verify chunk outputs are combined via Concat along axis=2
// UNROLLED:     IE.Concat({{%.+}}, {{%.+}}) {per_axis = #IE.Concat<axis = 2 : i64>}
// UNROLLED-SAME:  -> tensor<1x1x2x2x2xf32>

// Verify final reshape back to original output shape [B,H,N,dV]
// UNROLLED:     IE.Reshape
// UNROLLED-SAME:  -> tensor<1x1x4x2xf32>

}

// -----

// This test verifies that ChunkGatedDeltaRuleLoop REJECTS loops where gate is negated inside
// the body (meaning raw gate values are positive → chunked exp(cumGate) would overflow).
// This is the QWen3.5 pattern: gate = dt*A (positive), negated before Exp in the body.

// NEGATED-LABEL: module @ChunkGDR_PositiveGateRejected
module @ChunkGDR_PositiveGateRejected {

// NEGATED: func.func @main(
func.func @main(
    %k: tensor<1x1x4x2xf32>,
    %q: tensor<1x1x4x2xf32>,
    %v: tensor<1x1x4x2xf32>,
    %gate: tensor<1x1x4xf32>,
    %beta: tensor<1x1x4xf32>,
    %state: tensor<1x1x2x2xf32>,
    %accum: tensor<1x1x4x2xf32>) -> (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>) {

  %0:2 = IE.Loop(%k, %q, %v, %gate, %beta, %state, %accum)
      : tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4xf32>, tensor<1x1x4xf32>, tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
      -> tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
      (num_iterations : 4 current_iter_index : 0 exec_cond_index : 2)
      slice_input_descs : [
          #IE.SliceInputPortMap<external_port_id = 0 : i64, internal_layer_id = 1 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 1 : i64, internal_layer_id = 2 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 2 : i64, internal_layer_id = 3 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 3 : i64, internal_layer_id = 4 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>,
          #IE.SliceInputPortMap<external_port_id = 4 : i64, internal_layer_id = 5 : i64, axis = 2 : i64, start = 0 : i64, stride = 1 : i64, part_size = 1 : i64, end = 3 : i64>
      ]
      invariant_input_descs : []
      feedback_input_descs : [
          #IE.MergedInputPortMap<external_port_id = 5 : i64, internal_layer_id = 6 : i64, body_input_index = 0 : i64>,
          #IE.MergedInputPortMap<external_port_id = 6 : i64, internal_layer_id = 7 : i64, body_input_index = 1 : i64>
      ]
      concat_output_descs : []
      invariant_output_descs : [
          #IE.InvariantOutputPortMap<external_port_id = 0 : i64, internal_layer_id = 0 : i64, iterations = -1 : i64>,
          #IE.InvariantOutputPortMap<external_port_id = 1 : i64, internal_layer_id = 1 : i64, iterations = -1 : i64>
      ]
      body_module : {
            ^bb0(%iter: tensor<si32>,
           %k_t: tensor<1x1x1x2xf32>,
           %q_t: tensor<1x1x1x2xf32>,
           %v_t: tensor<1x1x1x2xf32>,
           %gate_t: tensor<1x1x1xf32>,
           %beta_t: tensor<1x1x1xf32>,
           %h_prev: tensor<1x1x2x2xf32>,
           %acc_prev: tensor<1x1x4x2xf32>):

        // Gate is NEGATED before Exp — raw gate values are positive (QWen3.5 pattern)
        %gate4 = IE.Reshape(%gate_t) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
        %neg_gate = IE.Negative(%gate4) : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
        %decay = IE.Exp(%neg_gate) : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>
        %h_decay = IE.Multiply(%h_prev, %decay) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x2x2xf32>

        %k_exp = IE.Reshape(%k_t) {shape_value = [1, 1, 2, 1]} : tensor<1x1x1x2xf32> -> tensor<1x1x2x1xf32>
        %hk_mul = IE.Multiply(%h_prev, %k_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x2xf32>
        %hk = IE.ReduceSum(%hk_mul) {axes_value = [2], keep_dims} : tensor<1x1x2x2xf32> -> tensor<1x1x1x2xf32>

        %delta = IE.Subtract(%v_t, %hk) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x2xf32>, tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf32>
        %beta4 = IE.Reshape(%beta_t) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
        %scaled = IE.Multiply(%delta, %beta4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x2xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x2xf32>

        %scaled_exp = IE.Reshape(%scaled) {shape_value = [1, 1, 1, 2]} : tensor<1x1x1x2xf32> -> tensor<1x1x1x2xf32>
        %upd = IE.Multiply(%k_exp, %scaled_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x1xf32>, tensor<1x1x1x2xf32> -> tensor<1x1x2x2xf32>
        %h_new = IE.Add(%h_decay, %upd) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32> -> tensor<1x1x2x2xf32>

        %q_exp = IE.Reshape(%q_t) {shape_value = [1, 1, 2, 1]} : tensor<1x1x1x2xf32> -> tensor<1x1x2x1xf32>
        %y_mul = IE.Multiply(%h_new, %q_exp) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x2x2xf32>, tensor<1x1x2x1xf32> -> tensor<1x1x2x2xf32>
        %y = IE.ReduceSum(%y_mul) {axes_value = [2], keep_dims} : tensor<1x1x2x2xf32> -> tensor<1x1x1x2xf32>

        %iter_index = IE.Reshape(%iter) {shape_value = [1]} : tensor<si32> -> tensor<1xsi32>
        %new_acc = IE.ScatterUpdate(%acc_prev, %iter_index, %y) {axis_value = 2 : i64} : tensor<1x1x4x2xf32>, tensor<1xsi32>, tensor<1x1x1x2xf32> -> tensor<1x1x4x2xf32>
        %cond = const.Declare tensor<1xi8> = dense<1> : tensor<1xi8>
        "IE.LoopTerminator"(%h_new, %new_acc, %cond) : (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>, tensor<1xi8>) -> ()
      }

  return %0#0, %0#1 : tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>
}

// Verify the Loop is NOT chunked (pass should skip it due to positive gate values)
// The original Loop with ScatterUpdate should remain unchanged
// NEGATED:     IE.Loop
// NEGATED:     IE.ScatterUpdate
// NEGATED-NOT: IE.CumSum

}
