//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/group_query_attention.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/op_conversions/group_query_attention_decomposition.hpp>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <cmath>

namespace ov::test {

namespace {

constexpr int32_t SEED = 42;

PRETTY_PARAM(NumHeads, int64_t);
PRETTY_PARAM(KVNumHeads, int64_t);
PRETTY_PARAM(SeqLen, int64_t);
PRETTY_PARAM(PastSeqLen, int64_t);
PRETTY_PARAM(HeadSize, int64_t);
PRETTY_PARAM(MaxSeqLen, int64_t);
PRETTY_PARAM(DoRotary, bool);

using GroupQueryAttentionParams = std::tuple<NumHeads, KVNumHeads, SeqLen, PastSeqLen, HeadSize, MaxSeqLen, DoRotary>;

class GroupQueryAttentionLayerTest :
        public VpuOv2LayerTest,
        public ::testing::WithParamInterface<GroupQueryAttentionParams> {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        const auto& [numHeadsParam, kvNumHeadsParam, seqLenParam, pastSeqLenParam, headSizeParam, maxSeqLenParam,
                     doRotaryParam] = GetParam();
        const auto seqLength = static_cast<size_t>(seqLenParam.value());
        const auto pastSeqLength = static_cast<size_t>(pastSeqLenParam.value());

        const auto& modelInputs = function->inputs();

        for (size_t i = 0; i < modelInputs.size(); i++) {
            const auto& input = modelInputs[i];
            const auto elemType = input.get_element_type();

            if (elemType == ov::element::i32) {
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<int32_t>();
                const auto& name = input.get_node_shared_ptr()->get_friendly_name();
                if (name == "seqlens_k") {
                    // Decomposition computes total_len = seqlens_k + 1,
                    // so seqlens_k = past + current - 1
                    data[0] = static_cast<int32_t>(pastSeqLength + seqLength - 1);
                } else if (name == "total_sequence_length") {
                    data[0] = static_cast<int32_t>(pastSeqLength + seqLength);
                }
                inputs.insert({input.get_node_shared_ptr(), tensor});
            } else {
                ov::test::utils::InputGenerateData genData(0, 1, 32, SEED);
                auto tensor = ov::test::utils::create_and_fill_tensor(elemType, targetInputStaticShapes[i], genData);
                inputs.insert({input.get_node_shared_ptr(), tensor});
            }
        }
    }

    void SetUp() override {
        VpuOv2LayerTest::SetUp();

        const auto& [numHeadsParam, kvNumHeadsParam, seqLenParam, pastSeqLenParam, headSizeParam, maxSeqLenParam,
                     doRotaryParam] = GetParam();

        const auto numHeads = static_cast<size_t>(numHeadsParam.value());
        const auto kvNumHeads = static_cast<size_t>(kvNumHeadsParam.value());
        const auto seqLength = static_cast<size_t>(seqLenParam.value());
        const auto pastSeqLength = static_cast<size_t>(pastSeqLenParam.value());
        const auto headSize = static_cast<size_t>(headSizeParam.value());
        const auto maxSeqLen = static_cast<size_t>(maxSeqLenParam.value());
        const auto rotary = doRotaryParam.value();
        const auto scale = 1.0f / std::sqrt(static_cast<float>(headSize));

        // past_key/past_value use pre-allocated buffer of size pastSeqLength + seqLength,
        // matching how real models work. GQA shape inference keeps output_kv_len =
        // past_key.shape[2] for static shapes, so past buffer must include room
        // for current tokens.
        const size_t totalSeqLen = pastSeqLength + seqLength;

        std::vector<ov::Shape> inputShapes;
        inputShapes.push_back({1, numHeads, seqLength, headSize});      // Q
        inputShapes.push_back({1, kvNumHeads, seqLength, headSize});    // K
        inputShapes.push_back({1, kvNumHeads, seqLength, headSize});    // V
        inputShapes.push_back({1, kvNumHeads, totalSeqLen, headSize});  // past_key
        inputShapes.push_back({1, kvNumHeads, totalSeqLen, headSize});  // past_value
        inputShapes.push_back({1, 1});                                  // seqlens_k
        inputShapes.push_back({1, 1});                                  // total_sequence_length

        if (rotary) {
            inputShapes.push_back({maxSeqLen, headSize / 2});  // cos_cache
            inputShapes.push_back({maxSeqLen, headSize / 2});  // sin_cache
        }

        init_input_shapes(ov::test::static_shapes_to_test_representation(inputShapes));

        auto makeParam = [](ov::element::Type type, const ov::Shape& shape, const std::string& name) {
            auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
            param->set_friendly_name(name);
            return param;
        };

        ov::ParameterVector params;
        params.push_back(makeParam(ov::element::f32, inputShapes[0], "Q"));
        params.push_back(makeParam(ov::element::f32, inputShapes[1], "K"));
        params.push_back(makeParam(ov::element::f32, inputShapes[2], "V"));
        params.push_back(makeParam(ov::element::f32, inputShapes[3], "past_key"));
        params.push_back(makeParam(ov::element::f32, inputShapes[4], "past_value"));
        params.push_back(makeParam(ov::element::i32, inputShapes[5], "seqlens_k"));
        params.push_back(makeParam(ov::element::i32, inputShapes[6], "total_sequence_length"));
        if (rotary) {
            params.push_back(makeParam(ov::element::f32, inputShapes[7], "cos_cache"));
            params.push_back(makeParam(ov::element::f32, inputShapes[8], "sin_cache"));
        }

        ov::OutputVector gqaInputs;
        for (auto& param : params) {
            gqaInputs.emplace_back(param);
        }

        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(gqaInputs, static_cast<int64_t>(numHeads),
                                                                           static_cast<int64_t>(kvNumHeads), scale,
                                                                           rotary, /*rotary_interleaved=*/false);
        gqa->set_friendly_name("gqa");

        ov::ResultVector results;
        for (size_t i = 0; i < gqa->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "GroupQueryAttention");

        // Decompose GQA into basic ops for CPU reference execution,
        // since GroupQueryAttention is an internal op without CPU kernel
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::GroupQueryAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }
};

TEST_P(GroupQueryAttentionLayerTest, NPU4000_HW) {
    configuration[ov::intel_npu::compilation_mode_params.name()] = "optimization-level=3";
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(GroupQueryAttentionLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

// MHA: num_heads == kv_num_heads, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_MHA_Rotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{8}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(PastSeqLen{64}),
                                            ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true})),
                         PrintTestCaseName());

// GQA: num_heads > kv_num_heads, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_GQA_Rotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(PastSeqLen{64}),
                                            ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true})),
                         PrintTestCaseName());

// GQA: without rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_GQA_NoRotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(PastSeqLen{64}),
                                            ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{false})),
                         PrintTestCaseName());

// MQA: kv_num_heads == 1, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_MQA_Rotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{8}), ::testing::Values(KVNumHeads{1}),
                                            ::testing::Values(SeqLen{4}), ::testing::Values(PastSeqLen{32}),
                                            ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true})),
                         PrintTestCaseName());

}  // namespace

}  // namespace ov::test
