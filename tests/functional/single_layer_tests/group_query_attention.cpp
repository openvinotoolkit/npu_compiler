//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/group_query_attention.hpp>
#include <openvino/op/transpose.hpp>
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
// Mistral-style local (sliding-window) attention: -1 disables the window (pure causal), a value
// >= 1 limits each query to attend to at most that many of the most recent keys.
PRETTY_PARAM(LocalWindowSize, int64_t);
PRETTY_PARAM(SlidingWindowCache, bool);

enum class Sink {
    NONE,
    SMOOTH_SOFTMAX,
    PER_HEAD,
};

PRETTY_PARAM(SinkMode, Sink);

// Some models store the V cache pre-transposed as [batch, kv_heads, head_size, seq] (matching a
// kernel-preferred matmul layout); a Transpose to/from [batch, kv_heads, seq, head_size] wraps the
// GQA op's past_value/present_value in that case.
PRETTY_PARAM(TransposedVCache, bool);

// attention_bias key width: CACHE_CAPACITY = KV cache length, MAX_CAPACITY = max sequence length.
// _2D broadcasts over heads [1, 1, seq, keys]; _3D is per head [1, num_heads, seq, keys].
enum class Bias {
    NONE,
    CACHE_CAPACITY_2D,
    MAX_CAPACITY_2D,
    CACHE_CAPACITY_3D,
    MAX_CAPACITY_3D,
};

PRETTY_PARAM(BiasMode, Bias);

using GroupQueryAttentionParams =
        std::tuple<NumHeads, KVNumHeads, SeqLen, SeqLenK, HeadSize, MaxSeqLen, DoRotary, KVCacheLen, KVCacheType,
                   SinkMode, LocalWindowSize, SlidingWindowCache, TransposedVCache, BiasMode>;

class GroupQueryAttentionLayerTest :
        public VpuOv2LayerTest,
        public ::testing::WithParamInterface<GroupQueryAttentionParams> {
protected:
    ov::element::Type kvCacheType_ = ov::element::f32;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        const auto& [numHeadsParam, kvNumHeadsParam, seqLenParam, seqLenKParam, headSizeParam, maxSeqLenParam,
                     doRotaryParam, kvCacheLenParam, kvCacheType, sinkModeParam, localWindowSizeParam,
                     slidingWindowCacheParam, transposedVCacheParam, biasModeParam] = GetParam();
        const auto seqLen = static_cast<size_t>(seqLenParam.value());
        const auto seqLenK = static_cast<size_t>(seqLenKParam.value());
        const auto headSize = static_cast<size_t>(headSizeParam.value());

        const auto& modelInputs = function->inputs();

        std::mt19937 rng(SEED);
        std::uniform_real_distribution<float> uniformDist(0.0f, 0.5f);

        constexpr float kUniformMax = 0.5f;
        constexpr float kSinkMargin = 3.0f;
        constexpr float kSinkPerHeadStep = 0.1f;
        const float sinkScale = 1.0f / std::sqrt(static_cast<float>(headSize));
        const float maxScoreBound = sinkScale * static_cast<float>(headSize) * kUniformMax * kUniformMax;
        const float sinkBase = maxScoreBound + kSinkMargin;

        for (size_t i = 0; i < modelInputs.size(); i++) {
            const auto& input = modelInputs[i];
            const auto elemType = input.get_element_type();
            const auto& name = input.get_node_shared_ptr()->get_friendly_name();

            if (name == "head_sink") {
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<float>();
                const auto totalElements = ov::shape_size(targetInputStaticShapes[i]);
                for (size_t idx = 0; idx < totalElements; idx++) {
                    data[idx] = sinkBase + static_cast<float>(idx) * kSinkPerHeadStep;
                }
                inputs.insert({input.get_node_shared_ptr(), tensor});
            } else if (name == "attention_bias") {
                // Wide spread so a few keys dominate the softmax; a small bias just averages out over random V.
                std::normal_distribution<float> biasDist(0.0f, 3.0f);
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                for (size_t idx = 0; idx < tensor.get_size(); idx++) {
                    const float value = biasDist(rng);
                    if (elemType == ov::element::f16) {
                        tensor.data<ov::float16>()[idx] = static_cast<ov::float16>(value);
                    } else {
                        tensor.data<float>()[idx] = value;
                    }
                }
                inputs.insert({input.get_node_shared_ptr(), tensor});
            } else if (elemType == ov::element::i32) {
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<int32_t>();
                if (name == "seqlens_k") {
                    // ONNX GQA convention: seqlens_k = (total tokens after this call) - 1, i.e. the zero-based
                    // index of the last valid token once the current seqLen tokens are appended to the
                    // seqLenK already-cached ones. Equals seqLenK for decode (seqLen == 1).
                    data[0] = static_cast<int32_t>(seqLenK + seqLen - 1);
                } else if (name == "total_sequence_length") {
                    // Decomposition scatters the new seqLen tokens at [seqlens_k, seqlens_k + seqLen);
                    // total_sequence_length must cover the whole new range, not just a single new token.
                    data[0] = static_cast<int32_t>(seqLenK + seqLen);
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
            } else if (elemType == ov::element::f16) {
                auto tensor = ov::Tensor(elemType, targetInputStaticShapes[i]);
                auto data = tensor.data<ov::float16>();
                const auto totalElements = ov::shape_size(targetInputStaticShapes[i]);
                for (size_t idx = 0; idx < totalElements; idx++) {
                    data[idx] = static_cast<ov::float16>(uniformDist(rng));
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
            // Outputs 1, 2 (present_key/present_value): relaxed only for an actually quantized (integral, e.g.
            // i8) cache; a plain reduced-precision float cache (e.g. f16) keeps the tight threshold.
            const double outAbsThreshold = (i > 0 && !kvCacheType_.is_real()) ? 1.0 : abs_threshold;

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
                     doRotaryParam, kvCacheLenParam, kvCacheTypeParam, sinkModeParam, localWindowSizeParam,
                     slidingWindowCacheParam, transposedVCacheParam, biasModeParam] = GetParam();

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
        const auto sinkMode = sinkModeParam.value();
        const auto smoothSoftmax = sinkMode == Sink::SMOOTH_SOFTMAX;
        const auto hasSink = sinkMode == Sink::PER_HEAD;
        const auto localWindowSize = localWindowSizeParam.value();
        const auto slidingWindowCache = slidingWindowCacheParam.value();
        const auto transposedVCache = transposedVCacheParam.value();
        const auto biasMode = biasModeParam.value();
        // A quantized (integral, e.g. i8) cache is handed to the operator as-is: it dequantizes/quantizes it
        // internally. A plain reduced-precision float cache (e.g. f16) needs no such handling.
        const auto quantizedKV = !kvCacheType.is_real();
        // The decomposition's non-quantized path scatter-updates past_key/past_value directly with the freshly
        // computed K/V, so Q/K/V must share the cache's element type there. The quantized path dequantizes/
        // quantizes via the operator's native K_SCALE/V_SCALE inputs, so Q/K/V stay f32.
        const auto computeType = quantizedKV ? ov::element::f32 : kvCacheType;

        using GQAInput = ov::op::internal::GroupQueryAttentionInputs;

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
        // Some models store the V cache pre-transposed as [batch, kv_heads, head_size, seq]; a Transpose
        // wraps past_value/present_value below to feed/consume the op in its expected layout.
        inputShapes.push_back(
                transposedVCache
                        ? ov::Shape{batchSize, static_cast<size_t>(kvNumHeads), headSize, totalSeqLen}
                        : ov::Shape{batchSize, static_cast<size_t>(kvNumHeads), totalSeqLen, headSize});  // past_value
        inputShapes.push_back({1, 1});                                                                    // seqlens_k
        inputShapes.push_back({1});  // total_sequence_length, op allows rank 0 or 1 only

        if (rotary) {
            inputShapes.push_back({maxSeqLen, headSize / 2});  // cos_cache
            inputShapes.push_back({maxSeqLen, headSize / 2});  // sin_cache
        }
        if (biasMode != Bias::NONE) {
            const bool cacheCapacity = biasMode == Bias::CACHE_CAPACITY_2D || biasMode == Bias::CACHE_CAPACITY_3D;
            const bool perHead = biasMode == Bias::CACHE_CAPACITY_3D || biasMode == Bias::MAX_CAPACITY_3D;
            const size_t biasLen = cacheCapacity ? totalSeqLen : maxSeqLen;
            const size_t biasHeads = perHead ? static_cast<size_t>(numHeads) : 1;
            inputShapes.push_back({1, biasHeads, seqLength, biasLen});  // attention_bias
        }
        if (hasSink) {
            inputShapes.push_back({static_cast<size_t>(numHeads)});  // head_sink
        }

        init_input_shapes(ov::test::static_shapes_to_test_representation(inputShapes));

        auto makeParam = [](ov::element::Type type, const ov::Shape& shape, const std::string& name) {
            auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
            param->set_friendly_name(name);
            return param;
        };

        ov::ParameterVector params;
        params.push_back(makeParam(computeType, inputShapes[0], "Q"));
        params.push_back(makeParam(computeType, inputShapes[1], "K"));
        params.push_back(makeParam(computeType, inputShapes[2], "V"));
        params.push_back(makeParam(kvCacheType, inputShapes[3], "past_key"));
        params.push_back(makeParam(kvCacheType, inputShapes[4], "past_value"));
        params.push_back(makeParam(ov::element::i32, inputShapes[5], "seqlens_k"));
        params.push_back(makeParam(ov::element::i32, inputShapes[6], "total_sequence_length"));
        if (rotary) {
            params.push_back(makeParam(computeType, inputShapes[7], "cos_cache"));
            params.push_back(makeParam(computeType, inputShapes[8], "sin_cache"));
        }
        std::shared_ptr<ov::op::v0::Parameter> biasParam;
        if (biasMode != Bias::NONE) {
            biasParam = makeParam(computeType, inputShapes[params.size()], "attention_bias");
            params.push_back(biasParam);
        }
        if (hasSink) {
            params.push_back(makeParam(computeType, inputShapes.back(), "head_sink"));
        }

        const auto vCacheTransposeOrder = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 3, 2});

        // For the plain reduced-precision (non-quantized) cache, the op itself computes in f32 here so the
        // CPU decomposition reference runs in f32 instead of emulated f16 (much faster); on the NPU path these
        // Converts are redundant (the whole graph gets lowered back to f16 regardless) and are expected to fold
        // away. Declared Parameter/Result types stay at kvCacheType, so the model's public interface is unchanged.
        const auto toComputeType = [&](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
            if (quantizedKV || value.get_element_type() == ov::element::f32) {
                return value;
            }
            return std::make_shared<ov::op::v0::Convert>(value, ov::element::f32);
        };

        ov::OutputVector gqaInputs;
        gqaInputs.emplace_back(toComputeType(params[0]));  // query
        gqaInputs.emplace_back(toComputeType(params[1]));  // key
        gqaInputs.emplace_back(toComputeType(params[2]));  // value
        gqaInputs.emplace_back(toComputeType(params[3]));  // past_key

        // past_value: undo the storage transpose into the op's expected [batch, kv_heads, seq, head_size].
        const ov::Output<ov::Node> pastValue =
                transposedVCache
                        ? ov::Output<ov::Node>(std::make_shared<ov::op::v1::Transpose>(params[4], vCacheTransposeOrder))
                        : ov::Output<ov::Node>(params[4]);
        gqaInputs.emplace_back(toComputeType(pastValue));  // past_value
        gqaInputs.emplace_back(params[5]);                 // seqlens_k
        gqaInputs.emplace_back(params[6]);                 // total_sequence_length

        // Optional inputs sit at fixed ONNX positions; skipped ones are padded with the empty-constant
        // placeholder meaning "absent".
        const auto appendOptional = [&](GQAInput position, const ov::Output<ov::Node>& value) {
            while (gqaInputs.size() < static_cast<size_t>(position)) {
                gqaInputs.emplace_back(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {}));
            }
            gqaInputs.emplace_back(value);
        };

        if (rotary) {
            appendOptional(GQAInput::COS_CACHE, toComputeType(params[7]));
            appendOptional(GQAInput::SIN_CACHE, toComputeType(params[8]));
        }
        if (biasParam) {
            appendOptional(GQAInput::ATTENTION_BIAS, toComputeType(biasParam));
        }
        if (hasSink) {
            appendOptional(GQAInput::HEAD_SINK, toComputeType(params.back()));
        }
        if (quantizedKV) {
            // Quantized KV cache: past_key/past_value stay integer, dequantized/requantized with
            // these scales. Per-channel scale, broadcastable over batch/seq_len: [1, kv_num_heads, 1, head_size].
            const float dqScaleVal = 0.01f;
            const ov::Shape dqScaleShape{1, static_cast<size_t>(kvNumHeads), 1, headSize};
            const std::vector<float> dqScaleData(shape_size(dqScaleShape), dqScaleVal);
            auto dqScale = ov::op::v0::Constant::create(ov::element::f32, dqScaleShape, dqScaleData);
            appendOptional(GQAInput::K_SCALE, dqScale);
            appendOptional(GQAInput::V_SCALE, dqScale);
        }

        const auto quantType = quantizedKV ? ov::op::internal::GroupQueryAttentionQuantType::PER_CHANNEL
                                           : ov::op::internal::GroupQueryAttentionQuantType::NONE;
        auto gqa = std::make_shared<ov::op::internal::GroupQueryAttention>(
                gqaInputs, numHeads, kvNumHeads, scale, rotary, /*rotary_interleaved=*/false,
                /*kv_cache_bit_width=*/quantizedKV ? 8 : 0, quantType, quantType, localWindowSize, slidingWindowCache,
                smoothSoftmax);
        gqa->set_friendly_name("gqa");

        // Output 0: attention output (f32); outputs 1, 2: present_key/present_value in the cache element type.
        // present_value is re-transposed to [batch, kv_heads, head_size, seq] to match past_value's storage
        // layout when the V cache is stored transposed.
        constexpr size_t presentKeyOutputIndex = 1;
        constexpr size_t presentValueOutputIndex = 2;
        ov::ResultVector results;
        for (size_t i = 0; i < gqa->get_output_size(); i++) {
            ov::Output<ov::Node> output = gqa->output(i);
            // present_key/present_value came out of the op in f32 (see toComputeType above); convert back
            // down to kvCacheType so the model's declared output type is unaffected.
            if (!quantizedKV && (i == presentKeyOutputIndex || i == presentValueOutputIndex)) {
                output = std::make_shared<ov::op::v0::Convert>(output, kvCacheType);
            }
            if (transposedVCache && i == presentValueOutputIndex) {
                output = std::make_shared<ov::op::v1::Transpose>(output, vCacheTransposeOrder);
            }
            results.push_back(std::make_shared<ov::op::v0::Result>(output));
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
INSTANTIATE_TEST_SUITE_P(
        smoke_MHA_Rotary, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{8}), ::testing::Values(KVNumHeads{8}),
                           ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                           ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                           ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                           ::testing::Values(KVCacheType{ov::element::f32}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

// GQA: num_heads > kv_num_heads, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(
        smoke_GQA_Rotary, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                           ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                           ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                           ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                           ::testing::Values(KVCacheType{ov::element::f32}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(
        smoke_GQA_Rotary_I8, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                           ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                           ::testing::Values(HeadSize{96}), ::testing::Values(MaxSeqLen{4096}),
                           ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                           ::testing::Values(KVCacheType{ov::element::i8}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

// GQA: without rotary embeddings
INSTANTIATE_TEST_SUITE_P(
        smoke_GQA_NoRotary, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                           ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                           ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                           ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{4096}),
                           ::testing::Values(KVCacheType{ov::element::f32}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

// GQA: without rotary embeddings
INSTANTIATE_TEST_SUITE_P(
        smoke_GQA_NoRotary_I8, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{32}), ::testing::Values(KVNumHeads{8}),
                           ::testing::Values(SeqLen{12}), ::testing::Values(SeqLenK{76}),
                           ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                           ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{4096}),
                           ::testing::Values(KVCacheType{ov::element::i8}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

// MQA: kv_num_heads == 1, with rotary embeddings
INSTANTIATE_TEST_SUITE_P(
        smoke_MQA_Rotary, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{8}), ::testing::Values(KVNumHeads{1}),
                           ::testing::Values(SeqLen{4}), ::testing::Values(SeqLenK{36}),
                           ::testing::Values(HeadSize{64}), ::testing::Values(MaxSeqLen{4096}),
                           ::testing::Values(DoRotary{true}), ::testing::Values(KVCacheLen{4096}),
                           ::testing::Values(KVCacheType{ov::element::f32}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

// FlashAttentionDMA kvcache test
INSTANTIATE_TEST_SUITE_P(
        smoke_attentionDmaFlashSmallSeqLength, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{24}),      // Q heads
                           ::testing::Values(KVNumHeads{8}),     // KV heads
                           ::testing::Values(SeqLen{2}),         // tsl
                           ::testing::Values(SeqLenK{11}),       // K/V sequence length
                           ::testing::Values(HeadSize{128}),     // e, eV
                           ::testing::Values(MaxSeqLen{8192}),   // max sequence length for rottary
                           ::testing::Values(DoRotary{false}),   // rotary embeddings
                           ::testing::Values(KVCacheLen{8192}),  // real SSL size
                           ::testing::Values(KVCacheType{ov::element::f32}), ::testing::Values(SinkMode{Sink::NONE}),
                           ::testing::Values(LocalWindowSize{-1}), ::testing::Values(SlidingWindowCache{false}),
                           ::testing::Values(TransposedVCache{false}), ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(
        smoke_attentionDmaFlash_24_8, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{24}), ::testing::Values(KVNumHeads{8}),
                           ::testing::Values(SeqLen{1}, SeqLen{8}, SeqLen{128}),
                           ::testing::Values(SeqLenK{7167}, SeqLenK{8191}), ::testing::Values(HeadSize{128}),
                           ::testing::Values(MaxSeqLen{32768}), ::testing::Values(DoRotary{false}),
                           ::testing::Values(KVCacheLen{8192}), ::testing::Values(KVCacheType{ov::element::f16}),
                           ::testing::Values(SinkMode{Sink::NONE}), ::testing::Values(LocalWindowSize{2047}),
                           ::testing::Values(SlidingWindowCache{true}), ::testing::Values(TransposedVCache{true}),
                           ::testing::Values(BiasMode{Bias::CACHE_CAPACITY_2D}, BiasMode{Bias::MAX_CAPACITY_2D})),
        PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(
        smoke_attentionDmaFlash_40_10, GroupQueryAttentionLayerTest,
        ::testing::Combine(::testing::Values(NumHeads{40}), ::testing::Values(KVNumHeads{10}),
                           ::testing::Values(SeqLen{1}, SeqLen{512}), ::testing::Values(SeqLenK{7167}, SeqLenK{8191}),
                           ::testing::Values(HeadSize{128}), ::testing::Values(MaxSeqLen{32768}),
                           ::testing::Values(DoRotary{false}), ::testing::Values(KVCacheLen{8192}),
                           ::testing::Values(KVCacheType{ov::element::f16}),
                           ::testing::Values(SinkMode{Sink::SMOOTH_SOFTMAX}), ::testing::Values(LocalWindowSize{4095}),
                           ::testing::Values(SlidingWindowCache{true}), ::testing::Values(TransposedVCache{true}),
                           ::testing::Values(BiasMode{Bias::NONE})),
        PrintTestCaseName());
}  // namespace

}  // namespace ov::test
