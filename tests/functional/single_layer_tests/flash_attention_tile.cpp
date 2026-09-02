//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_npu/ops/flash_attention_tile.hpp>

#include <openvino/core/type/float16.hpp>

#include <common/print_test_case_name.hpp>
#include <common/tensor_comparison.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <limits>
#include <memory>
#include <random>

namespace ov::test {

namespace {

const int32_t SEED = 42;

PRETTY_PARAM(Heads, BoundedDim);            // H
PRETTY_PARAM(QHeads, BoundedDim);           // H_q (for GQA)
PRETTY_PARAM(KVHeads, BoundedDim);          // H_kv (for GQA)
PRETTY_PARAM(SourceSeqLen, BoundedDim);     // S
PRETTY_PARAM(TargetSeqLen, BoundedDim);     // L
PRETTY_PARAM(QKEmbeddingSize, BoundedDim);  // E
PRETTY_PARAM(VEmbeddingSize, BoundedDim);   // Ev

// Absent          - no attention mask
// Broadcasted     - [1, 1, L, S]
// ZeroBroadcasted - [1, 1, L, S], filled with zeros
// Full            - [1, H, L, S]
enum struct Mask { Absent, Broadcasted, ZeroBroadcasted, Full };
PRETTY_PARAM(AttentionMask, Mask);

PRETTY_PARAM(IsHead, bool);
PRETTY_PARAM(IsTail, bool);

ov::ParameterVector generateInputParams(const std::vector<ov::PartialShape>& inputDynamicShapes,
                                        AttentionMask attentionMask) {
    ov::ParameterVector inputParams;
    const auto requiredParamNames =
            std::array<std::string_view, 6>{"query", "key", "value", "running_output", "running_max", "running_sum"};

    const auto inputType = ov::element::f16;
    for (auto [i, paramName] : vpux::enumerate(requiredParamNames)) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[i]));
        inputParams[i]->set_friendly_name(std::string{paramName});
    }

    const auto attentionMaskIdx = 6;
    if (attentionMask.value() != Mask::Absent) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[attentionMaskIdx]));
        inputParams.back()->set_friendly_name("attention_mask");
    }

    return inputParams;
}

// FlashSDPA MM1 (Attn*V) normalization constant. Must match FLASH_SDPA_MM1_NORM in
// sw_runtime_kernels/kernels/src/flash_sdpa.cpp: the kernel scales the second matmul
// (Attn*V) output by 1/FLASH_SDPA_MM1_NORM via the DPU PPE and compensates it only in
// the final tail divide. Therefore the running "Output" of a non-tail tile is stored at
// 1/FLASH_SDPA_MM1_NORM of its mathematical value, while the op reference produces the
// unnormalized running output. The tail output (and Max/Sum for all tiles) are unaffected.
constexpr float FLASH_SDPA_MM1_NORM = 32.0f;

// Return a copy of `src` with every element multiplied by `factor` (f16 or f32).
ov::Tensor scaleTensorCopy(const ov::Tensor& src, float factor) {
    ov::Tensor dst(src.get_element_type(), src.get_shape());
    const size_t n = src.get_size();
    if (src.get_element_type() == ov::element::f16) {
        const auto* s = src.data<ov::float16>();
        auto* d = dst.data<ov::float16>();
        for (size_t i = 0; i < n; ++i) {
            d[i] = ov::float16(static_cast<float>(s[i]) * factor);
        }
    } else {
        if (src.get_element_type() == ov::element::f32) {
            const auto* s = src.data<float>();
            auto* d = dst.data<float>();
            for (size_t i = 0; i < n; ++i) {
                d[i] = s[i] * factor;
            }
        } else {
            OPENVINO_ASSERT(false, "scaleTensorCopy supports only f16/f32 tensors");
        }
    }
    return dst;
}

template <typename Derived>
class FlashAttentionTileLayerTestBase : public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();

        std::mt19937 rng{SEED};

        const auto& modelInputs = function->inputs();
        const auto hasAttentionMask = (attentionMask != Mask::Absent);

        // Generate QKV
        for (int i = 0; i < 3; i++) {
            const auto inputType = modelInputs[i].get_element_type();
            ov::test::utils::InputGenerateData data(-1, 2, 32, SEED);
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(inputType, targetInputStaticShapes[i], data);
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }

        // Generate initial running out/max/sum constants
        for (int i = 3; i < 6; i++) {
            const auto inputType = modelInputs[i].get_element_type();
            const auto isRunningMaxTensor = (i == 4);

            auto tensor = ov::Tensor(inputType, targetInputStaticShapes[i]);
            if (inputType == ov::element::f16) {
                const ov::float16 tensorValue =
                        isRunningMaxTensor ? ov::float16(-std::numeric_limits<float>::infinity()) : ov::float16(0);
                auto data = tensor.data<ov::float16>();
                std::fill_n(data, tensor.get_size(), tensorValue);
            } else {
                const auto tensorValue = isRunningMaxTensor ? -std::numeric_limits<float>::infinity() : 0.f;
                auto data = tensor.data<float>();
                std::fill_n(data, tensor.get_size(), tensorValue);
            }

            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }

        // Generate attention mask
        if (hasAttentionMask) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            const auto sparsity = 0.3f;
            const auto isAllZeroMask = (attentionMask == Mask::ZeroBroadcasted);

            const auto attentionMaskIdx = 6;

            const auto inputType = modelInputs[attentionMaskIdx].get_element_type();
            auto tensor = ov::Tensor(inputType, targetInputStaticShapes[attentionMaskIdx]);
            const auto size = tensor.get_size();

            if (inputType == ov::element::f16) {
                auto data = tensor.data<ov::float16>();
                if (isAllZeroMask) {
                    std::fill_n(data, size, ov::float16(0));
                } else {
                    for (size_t i = 0; i < size; i++) {
                        data[i] = dist(rng) < sparsity ? ov::float16(-std::numeric_limits<float>::infinity())
                                                       : ov::float16(0);
                    }
                }
            } else {
                auto data = tensor.data<float>();
                if (isAllZeroMask) {
                    std::fill_n(data, size, 0.0f);
                } else {
                    for (size_t i = 0; i < size; i++) {
                        data[i] = dist(rng) < sparsity ? -std::numeric_limits<float>::infinity() : 0.0f;
                    }
                }
            }

            inputs.insert({modelInputs[attentionMaskIdx].get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        init_thresholds();

        ASSERT_EQ(expected.size(), 3u);
        ASSERT_EQ(actual.size(), 3u);

        const std::array<std::string_view, 3> names = {"Output", "Max", "Sum"};
        std::ostringstream failures;
        auto failureCount = 0;

        const auto isTail = this->isTail;

        // For non-tail tiles the device stores the running "Output" (index 0) scaled by
        // 1/FLASH_SDPA_MM1_NORM (the kernel compensates only in the tail divide), whereas
        // the op reference is unnormalized. Undo that scaling on the device output before
        // comparing. Max/Sum and the tail output need no adjustment.
        std::vector<ov::Tensor> cmpActual{actual.begin(), actual.end()};
        if (!isTail) {
            cmpActual[0] = scaleTensorCopy(actual[0], FLASH_SDPA_MM1_NORM);
        }

        auto check = [&](size_t i, double absThreshold, double relThreshold) {
            auto result = ov::test::utils::compareTensors(expected[i], cmpActual[i], absThreshold, relThreshold);

            auto warning = ov::test::utils::formatExpectedAnomalyWarning(result, names[i]);
            if (!warning.empty()) {
                vpux::Logger::global().warning("{0}", warning);
            }

            if (!result.passed()) {
                ++failureCount;
                failures << "\n" << ov::test::utils::formatComparisonResult(result, names[i]);
            }
        };

        if (isTail) {
            check(0, abs_threshold, rel_threshold);
        } else {
            check(0, abs_threshold, 5.0 * rel_threshold);
            check(1, abs_threshold, rel_threshold);
            check(2, abs_threshold, rel_threshold);
        }

        ASSERT_EQ(failureCount, 0) << failures.str();
    }

    void SetUp() override {
        VpuOv2LayerTest::SetUp();

        auto params = static_cast<Derived*>(this)->GetParam();

        this->attentionMask = std::get<AttentionMask>(params).value();
        this->isHead = std::get<IsHead>(params).value();
        this->isTail = std::get<IsTail>(params).value();

        const auto inputShapes = static_cast<Derived*>(this)->generateInputShapes(params);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, attentionMask);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto config = ov::intel_npu::op::FlashAttentionTile::Config();
        config.is_head = std::get<IsHead>(params).value();
        config.is_tail = std::get<IsTail>(params).value();

        auto flashAttentionTile = std::make_shared<ov::intel_npu::op::FlashAttentionTile>(inputs, config);
        flashAttentionTile->set_friendly_name("flash_attention_tile");

        ov::OutputVector outputs;
        for (const auto& output : flashAttentionTile->outputs()) {
            outputs.push_back(output);
        }

        function = std::make_shared<ov::Model>(outputs, inputParams, "FlashAttentionTile");
    }

public:
    Mask attentionMask = Mask::Absent;
    bool isHead = true;
    bool isTail = true;
};

using FlashAttentionTileParams =
        std::tuple<Heads, SourceSeqLen, TargetSeqLen, QKEmbeddingSize, VEmbeddingSize, AttentionMask, IsHead, IsTail>;

class FlashAttentionTileMHALayerTest :
        public FlashAttentionTileLayerTestBase<FlashAttentionTileMHALayerTest>,
        public ::testing::WithParamInterface<FlashAttentionTileParams> {
public:
    std::vector<InputShape> generateInputShapes(FlashAttentionTileParams params) {
        const auto& [heads, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, attentionMask, isHead,
                     isTail] = params;

        auto qShape = generateTestShape(heads.value(), targetSeqLen.value(), qkEmbeddingSize.value());
        auto kShape = generateTestShape(heads.value(), sourceSeqLen.value(), qkEmbeddingSize.value());
        auto vShape = generateTestShape(heads.value(), sourceSeqLen.value(), vEmbeddingSize.value());

        auto runningOutShape = generateTestShape(heads.value(), targetSeqLen.value(), vEmbeddingSize.value());
        auto runningMaxShape = generateTestShape(heads.value(), targetSeqLen.value());
        auto runningSumShape = generateTestShape(heads.value(), targetSeqLen.value());

        auto inputShapes =
                std::vector<InputShape>{qShape, kShape, vShape, runningOutShape, runningMaxShape, runningSumShape};

        if (attentionMask.value() == Mask::Full) {
            auto attentionMaskShape = generateTestShape(heads.value(), targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        } else if (attentionMask.value() == Mask::Broadcasted || attentionMask.value() == Mask::ZeroBroadcasted) {
            auto attentionMaskShape = generateTestShape(1, targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        }

        return inputShapes;
    }
};

using FlashAttentionTileGQAParams = std::tuple<QHeads, KVHeads, SourceSeqLen, TargetSeqLen, QKEmbeddingSize,
                                               VEmbeddingSize, AttentionMask, IsHead, IsTail>;

class FlashAttentionTileGQALayerTest :
        public FlashAttentionTileLayerTestBase<FlashAttentionTileGQALayerTest>,
        public ::testing::WithParamInterface<FlashAttentionTileGQAParams> {
public:
    std::vector<InputShape> generateInputShapes(FlashAttentionTileGQAParams params) {
        const auto& [qHeads, kvHeads, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, attentionMask,
                     isHead, isTail] = params;

        auto qShape = generateTestShape(qHeads.value(), targetSeqLen.value(), qkEmbeddingSize.value());
        auto kShape = generateTestShape(kvHeads.value(), sourceSeqLen.value(), qkEmbeddingSize.value());
        auto vShape = generateTestShape(kvHeads.value(), sourceSeqLen.value(), vEmbeddingSize.value());

        auto runningOutShape = generateTestShape(qHeads.value(), targetSeqLen.value(), vEmbeddingSize.value());
        auto runningMaxShape = generateTestShape(qHeads.value(), targetSeqLen.value());
        auto runningSumShape = generateTestShape(qHeads.value(), targetSeqLen.value());

        auto inputShapes =
                std::vector<InputShape>{qShape, kShape, vShape, runningOutShape, runningMaxShape, runningSumShape};

        if (attentionMask.value() == Mask::Full) {
            auto attentionMaskShape = generateTestShape(qHeads.value(), targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        } else if (attentionMask.value() == Mask::Broadcasted || attentionMask.value() == Mask::ZeroBroadcasted) {
            auto attentionMaskShape = generateTestShape(1, targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        }

        return inputShapes;
    }
};

}  // namespace

//
// FlashAttentionTileMHALayerTest
//

TEST_P(FlashAttentionTileMHALayerTest, NPU5010_HW) {
    abs_threshold = 0.001;
    rel_threshold = 0.001;
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke, FlashAttentionTileMHALayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                                           //
                                            ::testing::Values(SourceSeqLen{160}),                                  //
                                            ::testing::Values(TargetSeqLen{64}),                                   //
                                            ::testing::Values(QKEmbeddingSize{128}),                               //
                                            ::testing::Values(VEmbeddingSize{128}),                                //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Full,             //
                                                                                           Mask::Broadcasted,      //
                                                                                           Mask::ZeroBroadcasted,  //
                                                                                           Mask::Absent}),         //
                                            ::testing::ValuesIn(std::vector<IsHead>{true, false}),                 //
                                            ::testing::ValuesIn(std::vector<IsTail>{true, false})                  //
                                            ),                                                                     //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_SeqLen960, FlashAttentionTileMHALayerTest,
                         ::testing::Combine(::testing::Values(Heads{32}),                                      //
                                            ::testing::Values(SourceSeqLen{960}),                              //
                                            ::testing::Values(TargetSeqLen{960}),                              //
                                            ::testing::Values(QKEmbeddingSize{128}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                            //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,  //
                                                                                           Mask::Absent}),     //
                                            ::testing::ValuesIn(std::vector<IsHead>{false}),                   //
                                            ::testing::ValuesIn(std::vector<IsTail>{true})                     //
                                            ),                                                                 //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_SeqLen1024, FlashAttentionTileMHALayerTest,
                         ::testing::Combine(::testing::Values(Heads{32}),                                      //
                                            ::testing::Values(SourceSeqLen{1024}),                             //
                                            ::testing::Values(TargetSeqLen{1024}),                             //
                                            ::testing::Values(QKEmbeddingSize{128}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                            //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,  //
                                                                                           Mask::Absent}),     //
                                            ::testing::ValuesIn(std::vector<IsHead>{false}),                   //
                                            ::testing::ValuesIn(std::vector<IsTail>{true})                     //
                                            ),                                                                 //
                         PrintTestCaseName());

//
// FlashAttentionTileGQALayerTest
//

TEST_P(FlashAttentionTileGQALayerTest, NPU5010_HW) {
    abs_threshold = 0.001;
    rel_threshold = 0.001;
    run(Platform::NPU5010);
}

INSTANTIATE_TEST_SUITE_P(smoke_OneTargetSeqLen, FlashAttentionTileGQALayerTest,
                         ::testing::Combine(::testing::Values(QHeads{32}),                                         //
                                            ::testing::Values(KVHeads{8}),                                         //
                                            ::testing::Values(SourceSeqLen{1024}),                                 //
                                            ::testing::Values(TargetSeqLen{1}),                                    //
                                            ::testing::Values(QKEmbeddingSize{128}),                               //
                                            ::testing::Values(VEmbeddingSize{128}),                                //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,      //
                                                                                           Mask::ZeroBroadcasted,  //
                                                                                           Mask::Absent}),         //
                                            ::testing::ValuesIn(std::vector<IsHead>{false}),                       //
                                            ::testing::ValuesIn(std::vector<IsTail>{true})                         //
                                            ),                                                                     //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_SeqLen960, FlashAttentionTileGQALayerTest,
                         ::testing::Combine(::testing::Values(QHeads{32}),                                     //
                                            ::testing::Values(KVHeads{8}),                                     //
                                            ::testing::Values(SourceSeqLen{960}),                              //
                                            ::testing::Values(TargetSeqLen{960}),                              //
                                            ::testing::Values(QKEmbeddingSize{128}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                            //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,  //
                                                                                           Mask::Absent}),     //
                                            ::testing::ValuesIn(std::vector<IsHead>{false}),                   //
                                            ::testing::ValuesIn(std::vector<IsTail>{true})                     //
                                            ),                                                                 //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_SeqLen1024, FlashAttentionTileGQALayerTest,
                         ::testing::Combine(::testing::Values(QHeads{32}),                                     //
                                            ::testing::Values(KVHeads{8}),                                     //
                                            ::testing::Values(SourceSeqLen{1024}),                             //
                                            ::testing::Values(TargetSeqLen{1024}),                             //
                                            ::testing::Values(QKEmbeddingSize{128}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                            //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,  //
                                                                                           Mask::Absent}),     //
                                            ::testing::ValuesIn(std::vector<IsHead>{false}),                   //
                                            ::testing::ValuesIn(std::vector<IsTail>{true})                     //
                                            ),                                                                 //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_GQA_KVSeqLen, FlashAttentionTileGQALayerTest,
                         ::testing::Combine(::testing::Values(QHeads{32}),  //
                                            ::testing::Values(KVHeads{8}),  //
                                            ::testing::ValuesIn(std::vector<SourceSeqLen>{1024, 2048, 4096, 7168,
                                                                                          8192}),    //
                                            ::testing::ValuesIn(std::vector<TargetSeqLen>{1, 512}),  //
                                            ::testing::Values(QKEmbeddingSize{128}),                 //
                                            ::testing::Values(VEmbeddingSize{128}),                  //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),     //
                                            ::testing::ValuesIn(std::vector<IsHead>{true, false}),   //
                                            ::testing::ValuesIn(std::vector<IsTail>{true, false})    //
                                            ),                                                       //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_Gemma_prefill, FlashAttentionTileGQALayerTest,
                         ::testing::Combine(::testing::Values(QHeads{16}),                 //
                                            ::testing::Values(KVHeads{1}),                 //
                                            ::testing::Values(SourceSeqLen{1024}),         //
                                            ::testing::Values(TargetSeqLen{1024}),         //
                                            ::testing::Values(QKEmbeddingSize{512}),       //
                                            ::testing::Values(VEmbeddingSize{512}),        //
                                            ::testing::Values(AttentionMask{Mask::Full}),  //
                                            ::testing::Values(IsHead{false}),              //
                                            ::testing::Values(IsTail{true})                //
                                            ),                                             //
                         PrintTestCaseName());

}  // namespace ov::test
