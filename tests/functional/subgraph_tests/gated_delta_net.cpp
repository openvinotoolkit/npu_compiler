//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>

#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::test {

using GatedDeltaNetSubgraphParams = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, ov::element::Type>;

class GatedDeltaNetSubgraphCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<GatedDeltaNetSubgraphParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatedDeltaNetSubgraphParams>& obj) {
        const auto& [batch, seq_len, qk_head_num, v_head_num, qk_head_size, v_head_size, prec] = obj.param;
        std::ostringstream result;
        result << "batch=" << batch << "_seqLen=" << seq_len << "_qkHeads=" << qk_head_num << "_vHeads=" << v_head_num
               << "_qkHeadSize=" << qk_head_size << "_vHeadSize=" << v_head_size << "_prec=" << prec;
        return result.str();
    }

protected:
    static std::shared_ptr<ov::Model> buildLoopedGDN(int32_t batch, int32_t seq_len, int32_t qk_head_num,
                                                     int32_t v_head_num, int32_t qk_head_size, int32_t v_head_size,
                                                     ov::element::Type dtype) {
        const ov::PartialShape qk_shape{batch, seq_len, qk_head_num, qk_head_size};
        const ov::PartialShape v_tensor_shape{batch, seq_len, v_head_num, v_head_size};
        const ov::PartialShape gv_shape{batch, seq_len, v_head_num};
        const ov::PartialShape h_shape{batch, v_head_num, qk_head_size, v_head_size};

        auto q = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
        auto k = std::make_shared<ov::op::v0::Parameter>(dtype, qk_shape);
        auto v = std::make_shared<ov::op::v0::Parameter>(dtype, v_tensor_shape);
        auto h0 = std::make_shared<ov::op::v0::Parameter>(dtype, h_shape);
        auto g = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);
        auto beta = std::make_shared<ov::op::v0::Parameter>(dtype, gv_shape);

        q->set_friendly_name("q");
        k->set_friendly_name("k");
        v->set_friendly_name("v");
        h0->set_friendly_name("h0");
        g->set_friendly_name("g");
        beta->set_friendly_name("beta");

        const bool need_head_repeat = (qk_head_num != v_head_num);

        auto repeat_qk_heads =
                [&](const ov::Output<ov::Node>& query,
                    const ov::Output<ov::Node>& key) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
            if (!need_head_repeat) {
                return {query, key};
            }

            const int64_t group_size = static_cast<int64_t>(v_head_num / qk_head_num);

            std::vector<int64_t> repeated_head_ids;
            repeated_head_ids.reserve(static_cast<size_t>(v_head_num));
            for (int64_t h = 0; h < static_cast<int64_t>(v_head_num); ++h) {
                repeated_head_ids.push_back(h / group_size);
            }

            auto gather_indices = ov::op::v0::Constant::create(ov::element::i64, {static_cast<size_t>(v_head_num)},
                                                               repeated_head_ids);
            auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

            auto repeated_q = std::make_shared<ov::op::v8::Gather>(query, gather_indices, gather_axis, 0);
            auto repeated_k = std::make_shared<ov::op::v8::Gather>(key, gather_indices, gather_axis, 0);

            auto repeated_shape = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{0, 0, static_cast<int64_t>(v_head_num), static_cast<int64_t>(qk_head_size)});
            auto repeated_q_reshape = std::make_shared<ov::op::v1::Reshape>(repeated_q, repeated_shape, true);
            auto repeated_k_reshape = std::make_shared<ov::op::v1::Reshape>(repeated_k, repeated_shape, true);
            return {repeated_q_reshape, repeated_k_reshape};
        };

        auto l2norm = [&](const ov::Output<ov::Node>& x) {
            auto sq = std::make_shared<ov::op::v1::Multiply>(x, x);
            auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
            auto sum = std::make_shared<ov::op::v1::ReduceSum>(sq, axis, true);
            auto eps = ov::op::v0::Constant::create(dtype, {}, {1e-6f});
            auto inv = std::make_shared<ov::op::v1::Divide>(
                    ov::op::v0::Constant::create(dtype, {}, {1.0f}),
                    std::make_shared<ov::op::v0::Sqrt>(std::make_shared<ov::op::v1::Add>(sum, eps)));
            return std::make_shared<ov::op::v1::Multiply>(x, inv);
        };

        ov::Output<ov::Node> q_for_attn = q;
        ov::Output<ov::Node> k_for_attn = k;
        ov::Output<ov::Node> v_for_attn = v;

        if (need_head_repeat) {
            auto flatten_q_shape = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(1),
                                         static_cast<int64_t>(-1)});
            auto flatten_v_shape = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(0), static_cast<int64_t>(0), static_cast<int64_t>(1),
                                         static_cast<int64_t>(-1)});

            auto q_flat = std::make_shared<ov::op::v1::Reshape>(q, flatten_q_shape, true);
            auto k_flat = std::make_shared<ov::op::v1::Reshape>(k, flatten_q_shape, true);
            auto v_flat = std::make_shared<ov::op::v1::Reshape>(v, flatten_v_shape, true);

            auto qkv_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{q_flat, k_flat, v_flat}, -1);
            auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
            auto split_lengths = ov::op::v0::Constant::create(
                    ov::element::i64, {3},
                    {static_cast<int64_t>(qk_head_num) * static_cast<int64_t>(qk_head_size),
                     static_cast<int64_t>(qk_head_num) * static_cast<int64_t>(qk_head_size),
                     static_cast<int64_t>(v_head_num) * static_cast<int64_t>(v_head_size)});
            auto qkv_split = std::make_shared<ov::op::v1::VariadicSplit>(qkv_concat, split_axis, split_lengths);

            auto q_shape = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(0), static_cast<int64_t>(0),
                                         static_cast<int64_t>(qk_head_num), static_cast<int64_t>(qk_head_size)});
            auto k_shape = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(0), static_cast<int64_t>(0),
                                         static_cast<int64_t>(qk_head_num), static_cast<int64_t>(qk_head_size)});
            auto v_shape_split = ov::op::v0::Constant::create(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(0), static_cast<int64_t>(0),
                                         static_cast<int64_t>(v_head_num), static_cast<int64_t>(v_head_size)});

            q_for_attn = std::make_shared<ov::op::v1::Reshape>(qkv_split->output(0), q_shape, true);
            k_for_attn = std::make_shared<ov::op::v1::Reshape>(qkv_split->output(1), k_shape, true);
            v_for_attn = std::make_shared<ov::op::v1::Reshape>(qkv_split->output(2), v_shape_split, true);

            std::tie(q_for_attn, k_for_attn) = repeat_qk_heads(q_for_attn, k_for_attn);
        }

        auto q_norm = l2norm(q_for_attn);
        auto k_norm = l2norm(k_for_attn);

        auto perm_b_h_s_d = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
        auto perm_bhs = ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 2, 1});
        auto q_norm_t = std::make_shared<ov::op::v1::Transpose>(q_norm, perm_b_h_s_d);

        auto shape_of_q = std::make_shared<ov::op::v3::ShapeOf>(q_for_attn);
        auto gather_q_perm_index = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});
        auto gather_0_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto gather_q_shape = std::make_shared<ov::op::v8::Gather>(shape_of_q, gather_q_perm_index, gather_0_axis, 0);
        auto gather_head_size_index = ov::op::v0::Constant::create(ov::element::i64, {}, {3});
        auto gather_head_size =
                std::make_shared<ov::op::v8::Gather>(gather_q_shape, gather_head_size_index, gather_0_axis, 0);
        auto q_d = std::make_shared<ov::op::v0::Convert>(gather_head_size, dtype);
        auto half = ov::op::v0::Constant::create(dtype, {}, {0.5f});
        auto q_scale = std::make_shared<ov::op::v1::Power>(q_d, half);

        auto q_scaled_t = std::make_shared<ov::op::v1::Divide>(q_norm_t, q_scale);
        auto k_norm_t = std::make_shared<ov::op::v1::Transpose>(k_norm, perm_b_h_s_d);
        auto v_t = std::make_shared<ov::op::v1::Transpose>(v_for_attn, perm_b_h_s_d);
        auto g_t = std::make_shared<ov::op::v1::Transpose>(g, perm_bhs);
        auto beta_t = std::make_shared<ov::op::v1::Transpose>(beta, perm_bhs);

        auto vt_shape = std::make_shared<ov::op::v3::ShapeOf>(v_t);
        auto core_attn_init =
                std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(dtype, {}, {0.0f}), vt_shape);

        auto timestep = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
        auto q_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1, -1});
        auto k_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1, -1});
        auto v_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1, -1});
        auto h_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, -1, -1});
        auto g_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1});
        auto beta_i_param = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, 1});
        auto core_attn_buf = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape{-1, -1, -1, -1});

        auto minus1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
        auto minus2 = ov::op::v0::Constant::create(ov::element::i32, {1}, {-2});
        auto axis_2 = ov::op::v0::Constant::create(ov::element::i32, {}, {2});

        auto q_s = std::make_shared<ov::op::v0::Squeeze>(q_i_param, axis_2);
        auto k_s = std::make_shared<ov::op::v0::Squeeze>(k_i_param, axis_2);
        auto v_s = std::make_shared<ov::op::v0::Squeeze>(v_i_param, axis_2);

        auto g_exp = std::make_shared<ov::op::v0::Exp>(g_i_param);
        auto g_exp_unsq = std::make_shared<ov::op::v0::Unsqueeze>(g_exp, minus1);
        auto h_decay = std::make_shared<ov::op::v1::Multiply>(h_param, g_exp_unsq);

        auto k_unsq = std::make_shared<ov::op::v0::Unsqueeze>(k_s, minus1);
        auto hk = std::make_shared<ov::op::v1::Multiply>(h_decay, k_unsq);
        auto v_prime = std::make_shared<ov::op::v1::ReduceSum>(hk, minus2, false);
        auto v_new = std::make_shared<ov::op::v1::Subtract>(v_s, v_prime);
        auto v_scaled = std::make_shared<ov::op::v1::Multiply>(v_new, beta_i_param);

        auto v_scaled_unsq = std::make_shared<ov::op::v0::Unsqueeze>(v_scaled, minus2);
        auto h_update = std::make_shared<ov::op::v1::Multiply>(k_unsq, v_scaled_unsq);
        auto h_res = std::make_shared<ov::op::v1::Add>(h_decay, h_update);

        auto q_unsq = std::make_shared<ov::op::v0::Unsqueeze>(q_s, minus1);
        auto hq = std::make_shared<ov::op::v1::Multiply>(h_res, q_unsq);
        auto o_step = std::make_shared<ov::op::v1::ReduceSum>(hq, minus2, true);

        auto timestep_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                timestep, ov::op::v0::Constant::create(ov::element::i32, {1}, {0}));
        auto core_buf_res = std::make_shared<ov::op::v3::ScatterUpdate>(core_attn_buf, timestep_unsq, o_step, axis_2);

        auto body_cond = ov::op::v0::Constant::create(ov::element::boolean, {1}, {true});
        auto body = std::make_shared<ov::Model>(ov::OutputVector{body_cond, h_res, core_buf_res},
                                                ov::ParameterVector{timestep, q_i_param, k_i_param, v_i_param, h_param,
                                                                    g_i_param, beta_i_param, core_attn_buf},
                                                "recurrent_body");

        auto t_index = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto trip_count_i64 = std::make_shared<ov::op::v8::Gather>(vt_shape, t_index, gather_axis);
        auto trip_count = std::make_shared<ov::op::v0::Convert>(trip_count_i64, ov::element::i32);

        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count,
                                                       ov::op::v0::Constant::create(ov::element::boolean, {1}, {true}));
        loop->set_function(body);
        loop->set_sliced_input(q_i_param, q_scaled_t, 0, 1, 1, -1, 2);
        loop->set_sliced_input(k_i_param, k_norm_t, 0, 1, 1, -1, 2);
        loop->set_sliced_input(v_i_param, v_t, 0, 1, 1, -1, 2);
        loop->set_sliced_input(g_i_param, g_t, 0, 1, 1, -1, 2);
        loop->set_sliced_input(beta_i_param, beta_t, 0, 1, 1, -1, 2);
        loop->set_merged_input(h_param, h0, h_res);
        loop->set_merged_input(core_attn_buf, core_attn_init, core_buf_res);
        loop->set_special_body_ports({0, 0});

        auto core_attn_final_b_h_s_d = loop->get_iter_value(core_buf_res, -1);
        auto h_final = loop->get_iter_value(h_res, -1);

        auto reshape_m1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto flat_core = std::make_shared<ov::op::v1::Reshape>(core_attn_final_b_h_s_d, reshape_m1, false);
        auto flat_h = std::make_shared<ov::op::v1::Reshape>(h_final, reshape_m1, false);
        auto packed_loop_outputs = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{flat_core, flat_h}, 0);

        auto core_shape = std::make_shared<ov::op::v3::ShapeOf>(v_t);
        auto reduce_axis0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto core_numel = std::make_shared<ov::op::v1::ReduceProd>(core_shape, reduce_axis0, true);
        auto state_shape = std::make_shared<ov::op::v3::ShapeOf>(h0);

        auto slice_end_inf = ov::op::v0::Constant::create(ov::element::i64, {1}, {LLONG_MAX});
        auto slice_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto slice_step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto slice_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

        auto core_slice = std::make_shared<ov::op::v8::Slice>(packed_loop_outputs, slice_start, core_numel, slice_step,
                                                              slice_axis);
        auto state_slice = std::make_shared<ov::op::v8::Slice>(packed_loop_outputs, core_numel, slice_end_inf,
                                                               slice_step, slice_axis);

        auto core_restored = std::make_shared<ov::op::v1::Reshape>(core_slice, core_shape, false);
        auto state_restored = std::make_shared<ov::op::v1::Reshape>(state_slice, state_shape, false);

        auto core_attn_final = std::make_shared<ov::op::v1::Transpose>(
                core_restored, ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));

        auto final_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, v_head_size});
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(core_attn_final, final_shape, false);
        return std::make_shared<ov::Model>(ov::OutputVector{reshaped, state_restored},
                                           ov::ParameterVector{q, k, v, h0, g, beta});
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); ++i) {
            const auto elemType = modelInputs[i].get_element_type();
            const auto genData = (i == 4) ? ov::test::utils::InputGenerateData(-1, 1, 1000, 1)
                                          : ov::test::utils::InputGenerateData(0.0, 1, 1000, 1);
            auto tensor = ov::test::utils::create_and_fill_tensor(elemType, targetInputStaticShapes[i], genData);
            inputs.emplace(modelInputs[i].get_node_shared_ptr(), std::move(tensor));
        }
    }

    void SetUp() override {
        const auto& [batch, seq_len, qk_head_num, v_head_num, qk_head_size, v_head_size, prec] = GetParam();

        inType = prec;
        if (prec != ov::element::f32) {
            configuration[ov::hint::inference_precision.name()] = prec;
        }
        abs_threshold = (prec == ov::element::f32) ? 0.0015f : 0.01f;

        function = buildLoopedGDN(static_cast<int32_t>(batch), static_cast<int32_t>(seq_len),
                                  static_cast<int32_t>(qk_head_num), static_cast<int32_t>(v_head_num),
                                  static_cast<int32_t>(qk_head_size), static_cast<int32_t>(v_head_size), prec);

        const auto S = ov::Dimension(seq_len);
        const std::vector<ov::PartialShape> bounded = {
                {batch, S, qk_head_num, qk_head_size},
                {batch, S, qk_head_num, qk_head_size},
                {batch, S, v_head_num, v_head_size},
                {batch, v_head_num, qk_head_size, v_head_size},
                {batch, S, v_head_num},
                {batch, S, v_head_num},
        };
        std::map<ov::Output<ov::Node>, ov::PartialShape> reshapeMap;
        for (size_t i = 0; i < function->inputs().size(); ++i) {
            reshapeMap[function->input(i)] = bounded[i];
        }
        function->reshape(reshapeMap);

        const auto b = static_cast<size_t>(batch), sm = static_cast<size_t>(seq_len);
        const auto qh = static_cast<size_t>(qk_head_num), vh = static_cast<size_t>(v_head_num);
        const auto qd = static_cast<size_t>(qk_head_size), vd = static_cast<size_t>(v_head_size);
        init_input_shapes({
                {bounded[0], {{b, sm, qh, qd}}},
                {bounded[1], {{b, sm, qh, qd}}},
                {bounded[2], {{b, sm, vh, vd}}},
                {bounded[3], {{b, vh, qd, vd}}},
                {bounded[4], {{b, sm, vh}}},
                {bounded[5], {{b, sm, vh}}},
        });
    }
};

class GatedDeltaNetSubgraphTest_NPU5010 : public GatedDeltaNetSubgraphCommon {};

TEST_P(GatedDeltaNetSubgraphTest_NPU5010, TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

const std::vector<GatedDeltaNetSubgraphParams> gatedDeltaNetSubgraphCases = {
        {1, 16, 2, 2, 16, 16, ov::element::f16},   {1, 16, 2, 2, 64, 128, ov::element::f16},
        {1, 16, 2, 4, 16, 16, ov::element::f16},   {2, 8, 2, 2, 16, 16, ov::element::f16},
        {1, 16, 2, 2, 128, 128, ov::element::f32},
};

INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNet, GatedDeltaNetSubgraphTest_NPU5010,
                         ::testing::ValuesIn(gatedDeltaNetSubgraphCases), GatedDeltaNetSubgraphCommon::getTestCaseName);

// Decode cases (seq_len=1): GDN fusion should be skipped, model runs with original Loop graph
const std::vector<GatedDeltaNetSubgraphParams> gatedDeltaNetDecodeSubgraphCases = {
        {1, 1, 2, 2, 16, 16, ov::element::f16},
        {1, 1, 2, 2, 64, 128, ov::element::f16},
        {1, 1, 2, 4, 16, 16, ov::element::f16},
};

INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNet_Decode, GatedDeltaNetSubgraphTest_NPU5010,
                         ::testing::ValuesIn(gatedDeltaNetDecodeSubgraphCases),
                         GatedDeltaNetSubgraphCommon::getTestCaseName);

}  // namespace ov::test
