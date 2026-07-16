//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// Functional test for ChunkGatedDeltaRuleLoop pass.
// Builds a Loop implementing the Gated Delta Rule (GDR) recurrence:
//   h_t = exp(g_t) * h_{t-1} + k_t ⊗ beta_t * (v_t − <h_{t-1}, k_t>)
//   y_t = <h_t, q_t>
// and verifies that after chunked rewriting, inference accuracy matches CPU reference.
//

#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/subtract.hpp"

#include <algorithm>
#include <cstdint>
#include <random>

using namespace ov;
using namespace element;

namespace ov::test {

class GatedDeltaRuleLoopTestCommon : public VpuOv2LayerTest {
public:
    void SetUp() override {
        // GDR dimensions (small for test): batch=1, heads=1, seqLen=8, dK=2, dV=2
        constexpr int64_t B = 1, H = 1, N = 8, dK = 2, dV = 2;
        constexpr int64_t axis = 2;  // slice along sequence axis

        inType = ov::element::f32;

        // External inputs
        auto k_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N, dK});       // key
        auto q_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N, dK});       // query
        auto v_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N, dV});       // value
        auto gate_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N});        // gate (log domain)
        auto beta_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N});        // beta / dt
        auto state_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, dK, dV});  // initial h state
        auto accum_param = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N, dV});   // output accumulator

        k_param->set_friendly_name("k");
        q_param->set_friendly_name("q");
        v_param->set_friendly_name("v");
        gate_param->set_friendly_name("gate");
        beta_param->set_friendly_name("beta");
        state_param->set_friendly_name("state");
        accum_param->set_friendly_name("accum");

        ov::ParameterVector exParams = {k_param, q_param, v_param, gate_param, beta_param, state_param, accum_param};
        init_input_shapes(static_shapes_to_test_representation(
                {{B, H, N, dK}, {B, H, N, dK}, {B, H, N, dV}, {B, H, N}, {B, H, N}, {B, H, dK, dV}, {B, H, N, dV}}));

        // Body parameters (sliced inputs get 1 along sequence axis)
        auto body_k = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, 1, dK});       // k_t
        auto body_q = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, 1, dK});       // q_t
        auto body_v = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, 1, dV});       // v_t
        auto body_gate = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, 1});        // gate_t
        auto body_beta = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, 1});        // beta_t
        auto body_state = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, dK, dV});  // h_{t-1}
        auto body_accum = std::make_shared<op::v0::Parameter>(f32, Shape{B, H, N, dV});   // accumulator
        auto body_iter = std::make_shared<op::v0::Parameter>(i64, Shape{1});              // current iteration

        // ---- GDR body ----
        // Step 1: decay = exp(gate_t), gate reshaped to [B,H,1,1] for broadcasting
        // Gate values are already negative at Loop boundary (standard Mamba2: gate = -softplus(A)*dt)
        auto gate_reshaped = std::make_shared<op::v1::Reshape>(
                body_gate, op::v0::Constant::create(i64, Shape{4}, std::vector<int64_t>{B, H, 1, 1}), false);
        auto decay = std::make_shared<op::v0::Exp>(gate_reshaped);

        // Step 2: h_decay = decay * h_{t-1}: [B,H,1,1] * [B,H,dK,dV] → [B,H,dK,dV]
        auto h_decay = std::make_shared<op::v1::Multiply>(body_state, decay);

        // Step 3: <h_{t-1}, k_t> via elementwise mul + ReduceSum
        // k_expanded: [B,H,1,dK] → [B,H,dK,1] for outer product
        auto k_expanded = std::make_shared<op::v1::Reshape>(
                body_k, op::v0::Constant::create(i64, Shape{4}, std::vector<int64_t>{B, H, dK, 1}), false);
        // hk_mul = h_{t-1} * k_expanded: [B,H,dK,dV] * [B,H,dK,1] → [B,H,dK,dV]
        auto hk_mul = std::make_shared<op::v1::Multiply>(body_state, k_expanded);
        // hk = ReduceSum(hk_mul, axis=2): [B,H,dK,dV] → [B,H,1,dV]
        auto hk = std::make_shared<op::v1::ReduceSum>(
                hk_mul, op::v0::Constant::create(i64, Shape{1}, std::vector<int64_t>{2}), true);

        // Step 4: delta = v_t - hk: [B,H,1,dV]
        auto delta = std::make_shared<op::v1::Subtract>(body_v, hk);

        // Step 5: scaled = delta * beta_t
        auto beta_reshaped = std::make_shared<op::v1::Reshape>(
                body_beta, op::v0::Constant::create(i64, Shape{4}, std::vector<int64_t>{B, H, 1, 1}), false);
        auto scaled = std::make_shared<op::v1::Multiply>(delta, beta_reshaped);

        // Step 6: update = k_expanded ⊗ scaled = outer product: [B,H,dK,1] * [B,H,1,dV] → [B,H,dK,dV]
        auto scaled_reshaped = std::make_shared<op::v1::Reshape>(
                scaled, op::v0::Constant::create(i64, Shape{4}, std::vector<int64_t>{B, H, 1, dV}), false);
        auto update = std::make_shared<op::v1::Multiply>(k_expanded, scaled_reshaped);

        // Step 7: h_new = h_decay + update: [B,H,dK,dV]
        auto h_new = std::make_shared<op::v1::Add>(h_decay, update);

        // Step 8: y_t = <h_new, q_t>
        auto q_expanded = std::make_shared<op::v1::Reshape>(
                body_q, op::v0::Constant::create(i64, Shape{4}, std::vector<int64_t>{B, H, dK, 1}), false);
        auto yq_mul = std::make_shared<op::v1::Multiply>(h_new, q_expanded);
        auto y_t = std::make_shared<op::v1::ReduceSum>(
                yq_mul, op::v0::Constant::create(i64, Shape{1}, std::vector<int64_t>{2}), true);

        // Step 9: ScatterUpdate to accumulate y_t at position iter
        auto axis_const = op::v0::Constant::create(i64, Shape{1}, std::vector<int64_t>{axis});
        auto iter_i32 = std::make_shared<op::v0::Convert>(body_iter, i32);
        auto new_accum = std::make_shared<op::v3::ScatterUpdate>(body_accum, iter_i32, y_t, axis_const);

        // Internal execution condition: always true
        auto body_cond = op::v0::Constant::create(boolean, Shape{1}, {true});

        // Body results: [h_new, new_accum, exec_cond]
        auto result_h = std::make_shared<op::v0::Result>(h_new);
        auto result_accum = std::make_shared<op::v0::Result>(new_accum);
        auto result_cond = std::make_shared<op::v0::Result>(body_cond);

        auto body_model = std::make_shared<Model>(
                OutputVector{result_h, result_accum, result_cond},
                ParameterVector{body_iter, body_k, body_q, body_v, body_gate, body_beta, body_state, body_accum});

        // ---- Create Loop ----
        auto trip_count = op::v0::Constant::create(i64, Shape{1}, std::vector<int64_t>{N});
        auto exec_cond = op::v0::Constant::create(boolean, Shape{1}, {true});

        auto loop = std::make_shared<op::v5::Loop>(trip_count, exec_cond);
        // current_iteration_input_idx=0, body_condition_output_idx=2
        loop->set_special_body_ports(op::v5::Loop::SpecialBodyPorts{0, 2});
        loop->set_function(body_model);

        // Sliced inputs: k, q, v (4D), gate, beta (3D) along axis=2
        loop->set_sliced_input(body_k, k_param, 0, 1, 1, -1, axis);
        loop->set_sliced_input(body_q, q_param, 0, 1, 1, -1, axis);
        loop->set_sliced_input(body_v, v_param, 0, 1, 1, -1, axis);
        loop->set_sliced_input(body_gate, gate_param, 0, 1, 1, -1, axis);
        loop->set_sliced_input(body_beta, beta_param, 0, 1, 1, -1, axis);

        // Feedback (merged) inputs: state and accumulator
        loop->set_merged_input(body_state, state_param, result_h);
        loop->set_merged_input(body_accum, accum_param, result_accum);

        // Outputs: final h_state and final accumulator (last iteration)
        auto out_state = loop->get_iter_value(result_h, -1);
        auto out_accum = loop->get_iter_value(result_accum, -1);

        function = std::make_shared<Model>(OutputVector{out_state, out_accum}, exParams, "GatedDeltaRuleLoop");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();

        // Use small random values for numerical stability
        for (size_t i = 0; i < modelInputs.size(); ++i) {
            const auto& inputType = modelInputs[i].get_element_type();
            auto tensor = ov::Tensor(inputType, targetInputStaticShapes[i]);
            auto* ptr = static_cast<float*>(tensor.data());
            const auto numElements = tensor.get_size();

            // Zero-initialize state and accumulator for clean start
            if (i == 5 || i == 6) {
                std::fill_n(ptr, numElements, 0.0f);
            } else {
                // Small random values seeded deterministically
                std::mt19937 rng(42 + static_cast<uint32_t>(i));
                std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
                for (size_t j = 0; j < numElements; ++j) {
                    ptr[j] = dist(rng);
                }
            }

            inputs.emplace(modelInputs[i].get_node_shared_ptr(), std::move(tensor));
        }
    }
};

// GDR Loop functional test: NPU compilation with ChunkGatedDeltaRuleLoop pass vs CPU reference
class GatedDeltaRuleLoopTest_NPU4000 : public GatedDeltaRuleLoopTestCommon {};
class GatedDeltaRuleLoopTest_NPU5010 : public GatedDeltaRuleLoopTestCommon {};

TEST_F(GatedDeltaRuleLoopTest_NPU4000, TestKindSubgraph) {
    abs_threshold = 1e-4;
    rel_threshold = 1e-3;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_F(GatedDeltaRuleLoopTest_NPU5010, TestKindSubgraph) {
    abs_threshold = 1e-4;
    rel_threshold = 1e-3;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace ov::test
