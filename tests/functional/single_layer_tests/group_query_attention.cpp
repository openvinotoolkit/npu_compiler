//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/type/element_type.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/group_query_attention.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/round.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/op_conversions/group_query_attention_decomposition.hpp>

#include <common/print_test_case_name.hpp>
#include <common/tensor_comparison.hpp>
#include <common_test_utils/ov_plugin_cache.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <common_test_utils/test_constants.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

namespace ov::test {

namespace {

const int32_t SEED = 42;

PRETTY_PARAM(NumHeads, int64_t);
PRETTY_PARAM(KVNumHeads, int64_t);
PRETTY_PARAM(SeqLen, int64_t);
PRETTY_PARAM(SeqLenK, int64_t);
PRETTY_PARAM(HeadSize, int64_t);
PRETTY_PARAM(MaxSeqLen, int64_t);
PRETTY_PARAM(DoRotary, bool);
PRETTY_PARAM(KVCacheLen, int64_t);
PRETTY_PARAM(KVCacheType, ov::element::Type);

using GroupQueryAttentionParams =
        std::tuple<NumHeads, KVNumHeads, SeqLen, SeqLenK, HeadSize, MaxSeqLen, DoRotary, KVCacheLen, KVCacheType>;

class GroupQueryAttentionLayerTest :
        public VpuOv2LayerTest,
        public ::testing::WithParamInterface<GroupQueryAttentionParams> {
protected:
    ov::element::Type kvCacheType_ = ov::element::f32;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        const auto& [numHeadsParam, kvNumHeadsParam, seqLenParam, seqLenKParam, headSizeParam, maxSeqLenParam,
                     doRotaryParam, kvCacheLenParam, kvCacheType] = GetParam();
        const auto seqLenK = static_cast<size_t>(seqLenKParam.value());

        const auto& modelInputs = function->inputs();

        std::mt19937 rng(SEED);
        std::uniform_real_distribution<float> uniformDist(0.0f, 0.5f);

        for (size_t i = 0; i < modelInputs.size(); i++) {
            const auto& input = modelInputs[i];
            const auto elemType = input.get_element_type();
            const auto& name = input.get_node_shared_ptr()->get_friendly_name();

            if (elemType == ov::element::i32) {
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<int32_t>();
                if (name == "seqlens_k") {
                    data[0] = static_cast<int32_t>(seqLenK);
                } else if (name == "total_sequence_length") {
                    // Decomposition computes total_len = seqlens_k + 1
                    data[0] = static_cast<int32_t>(seqLenK + 1);
                }
                inputs.insert({input.get_node_shared_ptr(), tensor});
            } else if (elemType == ov::element::i8) {
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<int8_t>();
                const auto totalElements = ov::shape_size(targetInputStaticShapes[i]);
                std::uniform_int_distribution<int> intDist(-128, 127);
                for (size_t idx = 0; idx < totalElements; idx++) {
                    data[idx] = static_cast<int8_t>(intDist(rng));
                }
                inputs.insert({input.get_node_shared_ptr(), tensor});
            } else {
                // Q, K, V, pastK, pastV: uniform distribution in [0, 0.5]
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<float>();
                const auto totalElements = ov::shape_size(targetInputStaticShapes[i]);
                for (size_t idx = 0; idx < totalElements; idx++) {
                    data[idx] = uniformDist(rng);
                }
                inputs.insert({input.get_node_shared_ptr(), tensor});
            }
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        init_thresholds();

        ASSERT_EQ(expected.size(), actual.size());

        std::ostringstream failures;
        auto failureCount = 0;

        const auto outNames = std::array<std::string_view, 3>{"attention", "present_key", "present_value"};

        for (size_t i = 0; i < expected.size(); i++) {
            // Output 0 (attention, f32): tight threshold
            // Outputs 1, 2 (present_key/present_value): relaxed for i8 quantization
            const double outAbsThreshold = (i > 0 && kvCacheType_ != ov::element::f32) ? 1.0 : abs_threshold;

            auto result = ov::test::utils::compareTensors(expected[i], actual[i], outAbsThreshold, rel_threshold);

            if (!result.passed()) {
                ++failureCount;
                failures << "\n" << ov::test::utils::formatComparisonResult(result, outNames[i]);
            }
        }

        ASSERT_EQ(failureCount, 0) << failures.str();
    }

    void SetUp() override {
        VpuOv2LayerTest::SetUp();

        const auto& [numHeadsParam, kvNumHeadsParam, seqLenParam, seqLenKParam, headSizeParam, maxSeqLenParam,
                     doRotaryParam, kvCacheLenParam, kvCacheTypeParam] = GetParam();

        const size_t batchSize = 1;
        const int64_t numHeads = numHeadsParam.value();
        const int64_t kvNumHeads = kvNumHeadsParam.value();
        const auto seqLength = static_cast<size_t>(seqLenParam.value());
        const auto seqLenK = static_cast<size_t>(seqLenKParam.value());
        const auto headSize = static_cast<size_t>(headSizeParam.value());
        const auto maxSeqLen = static_cast<size_t>(maxSeqLenParam.value());
        const auto rotary = doRotaryParam.value();
        const auto scale = 1.0f / std::sqrt(static_cast<float>(headSize));
        const auto kvCacheType = kvCacheTypeParam.value();

        kvCacheType_ = kvCacheType;

        // past_key/past_value use pre-allocated buffer. When kvCacheLen > 0, use it
        // to simulate a large pre-allocated KV cache (e.g., 8K) with few valid tokens.
        // Otherwise the buffer exactly fits seqLenK tokens.
        const auto kvCacheLen = static_cast<size_t>(kvCacheLenParam.value());
        const size_t totalSeqLen = kvCacheLen > 0 ? kvCacheLen : seqLenK;

        std::vector<ov::Shape> inputShapes;
        inputShapes.push_back({batchSize, static_cast<size_t>(numHeads), seqLength, headSize});      // Q
        inputShapes.push_back({batchSize, static_cast<size_t>(kvNumHeads), seqLength, headSize});    // K
        inputShapes.push_back({batchSize, static_cast<size_t>(kvNumHeads), seqLength, headSize});    // V
        inputShapes.push_back({batchSize, static_cast<size_t>(kvNumHeads), totalSeqLen, headSize});  // past_key
        inputShapes.push_back({batchSize, static_cast<size_t>(kvNumHeads), totalSeqLen, headSize});  // past_value
        inputShapes.push_back({1, 1});                                                               // seqlens_k
        inputShapes.push_back({1, 1});  // total_sequence_length

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
        params.push_back(makeParam(kvCacheType, inputShapes[3], "past_key"));
        params.push_back(makeParam(kvCacheType, inputShapes[4], "past_value"));
        params.push_back(makeParam(ov::element::i32, inputShapes[5], "seqlens_k"));
        params.push_back(makeParam(ov::element::i32, inputShapes[6], "total_sequence_length"));
        if (rotary) {
            params.push_back(makeParam(ov::element::f32, inputShapes[7], "cos_cache"));
            params.push_back(makeParam(ov::element::f32, inputShapes[8], "sin_cache"));
        }

        ov::OutputVector gqaInputs;
        // Q, K, V pass through as f32
        gqaInputs.emplace_back(params[0]);  // Q
        gqaInputs.emplace_back(params[1]);  // K
        gqaInputs.emplace_back(params[2]);  // V

        if (kvCacheType != ov::element::f32) {
            // Dequantize past_key/past_value: Convert(i8→f32) → Multiply(scale)
            const float dqScaleVal = 0.01f;
            auto dqScale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {dqScaleVal});

            auto pastKeyCvt = std::make_shared<ov::op::v0::Convert>(params[3], ov::element::f32);
            auto pastKeyDq =
                    std::make_shared<ov::op::v1::Multiply>(pastKeyCvt, dqScale, ov::op::AutoBroadcastType::NUMPY);

            auto pastValCvt = std::make_shared<ov::op::v0::Convert>(params[4], ov::element::f32);
            auto pastValDq =
                    std::make_shared<ov::op::v1::Multiply>(pastValCvt, dqScale, ov::op::AutoBroadcastType::NUMPY);

            gqaInputs.emplace_back(pastKeyDq);
            gqaInputs.emplace_back(pastValDq);
        } else {
            gqaInputs.emplace_back(params[3]);  // past_key f32
            gqaInputs.emplace_back(params[4]);  // past_value f32
        }

        // seqlens_k, total_sequence_length, [cos_cache, sin_cache]
        for (size_t i = 5; i < params.size(); i++) {
            gqaInputs.emplace_back(params[i]);
        }

        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(gqaInputs, numHeads, kvNumHeads, scale,
                                                                           rotary, /*rotary_interleaved=*/false);
        gqa->set_friendly_name("gqa");

        ov::ResultVector results;
        // Output 0: attention output (f32)
        results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(0)));

        if (kvCacheType != ov::element::f32 && gqa->get_output_size() > 1) {
            // Quantize present_key/present_value:
            // Multiply(1/scale) → Round(HALF_TO_EVEN) → Clamp(-128, 127) → Convert(f32→i8)
            const float dqScaleVal = 0.01f;
            auto invScale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / dqScaleVal});

            for (size_t i = 1; i < gqa->get_output_size(); i++) {
                auto mul = std::make_shared<ov::op::v1::Multiply>(gqa->output(i), invScale,
                                                                  ov::op::AutoBroadcastType::NUMPY);
                auto round = std::make_shared<ov::op::v5::Round>(mul, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
                auto clamp = std::make_shared<ov::op::v0::Clamp>(round, -128.0, 127.0);
                auto cvt = std::make_shared<ov::op::v0::Convert>(clamp, kvCacheType);
                results.push_back(std::make_shared<ov::op::v0::Result>(cvt));
            }
        } else {
            for (size_t i = 1; i < gqa->get_output_size(); i++) {
                results.push_back(std::make_shared<ov::op::v0::Result>(gqa->output(i)));
            }
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
                                            ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                                            ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

// GQA: num_heads > kv_num_heads, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_GQA_Rotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                                            ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_GQA_Rotary_I8, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                                            ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                                            ::testing::Values(KVCacheType{ov::element::i8})),
                         PrintTestCaseName());

// GQA: without rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_GQA_NoRotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                                            ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{4096}),
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

// GQA: without rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_GQA_NoRotary_I8, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                                            ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                                            ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{4096}),
                                            ::testing::Values(KVCacheType{ov::element::i8})),
                         PrintTestCaseName());

// MQA: kv_num_heads == 1, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(smoke_MQA_Rotary, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{8}), ::testing::Values(KVNumHeads{1}),
                                            ::testing::Values(SeqLen{4}), ::testing::Values(SeqLenK{36}),
                                            ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                                            ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

// FlashAttentionDMA prefill test
INSTANTIATE_TEST_SUITE_P(smoke_FADMA_Prefill, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{40}), ::testing::Values(KVNumHeads{10}),
                                            ::testing::Values(SeqLen{512}),
                                            ::testing::Values(SeqLenK{512}, SeqLenK{1535}, SeqLenK{4095},
                                                              SeqLenK{8191}),

                                            ::testing::Values(HeadSize{128}), ::testing::Values(MaxSeqLen{8192}),
                                            ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{8192}),
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

// FlashAttentionDMA kvcache test
INSTANTIATE_TEST_SUITE_P(smoke_FADMA_KVCache, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{40}), ::testing::Values(KVNumHeads{10}),
                                            ::testing::Values(SeqLen{1}),
                                            ::testing::Values(SeqLenK{511}, SeqLenK{1535}, SeqLenK{4095},
                                                              SeqLenK{8191}),
                                            ::testing::Values(HeadSize{128}), ::testing::Values(MaxSeqLen{8192}),
                                            ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{8192}),
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

// FlashAttentionDMA kvcache test
INSTANTIATE_TEST_SUITE_P(smoke_attentionDmaFlashSmallSeqLength, GroupQueryAttentionLayerTest,
                         ::testing::Combine(::testing::Values(NumHeads{24}),      // Q heads
                                            ::testing::Values(KVNumHeads{8}),     // KV heads
                                            ::testing::Values(SeqLen{2}),         // tsl
                                            ::testing::Values(SeqLenK{11}),       // K/V sequence length
                                            ::testing::Values(HeadSize{128}),     // e, eV
                                            ::testing::Values(MaxSeqLen{8192}),   // max sequence length for rottary
                                            ::testing::Values(DoRotary{false}),   // rotary embeddings
                                            ::testing::Values(KVCacheLen{8192}),  // real SSL size
                                            ::testing::Values(KVCacheType{ov::element::f32})),
                         PrintTestCaseName());

}  // namespace

}  // namespace ov::test
