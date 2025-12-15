//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/opsets/opset14_decl.hpp>

#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>
#include <vpux/utils/core/env.hpp>

namespace ov::test {

namespace {

// Cross Attention Params
// Batch == 1
PRETTY_PARAM(Heads, BoundedDim);            // H
PRETTY_PARAM(SourceSeqLen, BoundedDim);     // S
PRETTY_PARAM(TargetSeqLen, BoundedDim);     // L
PRETTY_PARAM(QKEmbeddingSize, BoundedDim);  // E
PRETTY_PARAM(VEmbeddingSize, BoundedDim);   // Ev
PRETTY_PARAM(IsCausal, bool);

// Absent      - no attention mask
// Broadcasted - [1, 1, L, S]
// Full        - [1, H, L, S]
enum struct Mask { Absent, Broadcasted, Full };
PRETTY_PARAM(AttentionMask, Mask);
PRETTY_PARAM(HasScale, bool);

// Self Attention Params
PRETTY_PARAM(SequenceLength, BoundedDim);  // S == L
PRETTY_PARAM(EmbeddingSize, BoundedDim);   // E == Ev

enum struct SDPA { Normal, FlashAttention };
PRETTY_PARAM(Kind, SDPA);

const auto ALL_KINDS = std::vector<Kind>{SDPA::Normal, SDPA::FlashAttention};

ov::ParameterVector generateInputParams(const std::vector<ov::PartialShape>& inputDynamicShapes,
                                        AttentionMask attentionMask, HasScale hasScale) {
    ov::ParameterVector inputParams;

    const auto inputType = ov::element::f32;
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[2]));

    inputParams[0]->set_friendly_name("query");
    inputParams[1]->set_friendly_name("key");
    inputParams[2]->set_friendly_name("value");

    if (hasScale.value()) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(
                inputType, (attentionMask.value() != Mask::Absent) ? inputDynamicShapes[3] : ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    } else if (attentionMask.value() != Mask::Absent) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[3]));
        inputParams.back()->set_friendly_name("attention_mask");
    }

    return inputParams;
}

template <typename Derived>
class SdpaLayerTestBase : public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        auto shapes = std::vector<ov::Shape>{targetInputStaticShapes[0], targetInputStaticShapes[1],
                                             targetInputStaticShapes[2]};
        if (hasScale) {
            shapes.push_back((attentionMask != Mask::Absent) ? targetInputStaticShapes[3] : ov::Shape{});
            shapes.push_back(ov::Shape{1});  // NOLINT emplace_back leads to incorrect results
        } else if (attentionMask != Mask::Absent) {
            shapes.push_back(targetInputStaticShapes[3]);
        }

        SubgraphBaseTest::generate_inputs(shapes);
    }

    void configure_model() override {
        auto params = static_cast<Derived*>(this)->GetParam();
        auto kind = std::get<Kind>(params);

        if (kind.value() == SDPA::FlashAttention) {
            // Enable a pass that performs SDPA conversion to FlashSDPA
            configuration[ov::intel_npu::compilation_mode_params.name()] =
                    "enable-decompose-sdpa=false enable-flash-sdpa-conversion=true";
        }
    }

    void SetUp() override {
        VpuOv2LayerTest::SetUp();

        auto params = static_cast<Derived*>(this)->GetParam();
        auto kind = std::get<Kind>(params);

        this->attentionMask = std::get<AttentionMask>(params).value();
        this->hasScale = std::get<HasScale>(params).value();

        const auto inputShapes = static_cast<Derived*>(this)->generateInputShapes(params);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, attentionMask, hasScale);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        const auto isCausal = std::get<IsCausal>(params).value();
        auto sdpa = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCausal);
        sdpa->set_friendly_name("sdpa");

        function = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, inputParams, "SDPA");
    }

    void TearDown() override {
        VpuOv2LayerTest::TearDown();

        auto params = static_cast<Derived*>(this)->GetParam();
        auto kind = std::get<Kind>(params);
    }

public:
    Mask attentionMask = Mask::Absent;
    bool hasScale = false;
};

using CrossMultiHeadAttentionParams = std::tuple<Heads, SourceSeqLen, TargetSeqLen, QKEmbeddingSize, VEmbeddingSize,
                                                 IsCausal, AttentionMask, HasScale, Kind>;

class CrossMultiHeadAttentionLayerTest :
        public SdpaLayerTestBase<CrossMultiHeadAttentionLayerTest>,
        public ::testing::WithParamInterface<CrossMultiHeadAttentionParams> {
public:
    std::vector<InputShape> generateInputShapes(CrossMultiHeadAttentionParams params) {
        const auto& [heads, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, isCausal, attentionMask,
                     hasScale, kind] = params;

        auto qShape = generateTestShape(heads.value(), targetSeqLen.value(), qkEmbeddingSize.value());
        auto kShape = generateTestShape(heads.value(), sourceSeqLen.value(), qkEmbeddingSize.value());
        auto vShape = generateTestShape(heads.value(), sourceSeqLen.value(), vEmbeddingSize.value());

        auto inputShapes = std::vector<InputShape>{qShape, kShape, vShape};

        if (attentionMask.value() == Mask::Full) {
            auto attentionMaskShape = generateTestShape(heads.value(), targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        } else if (attentionMask.value() == Mask::Broadcasted) {
            auto attentionMaskShape = generateTestShape(1, targetSeqLen.value(), sourceSeqLen.value());
            inputShapes.push_back(attentionMaskShape);
        }

        return inputShapes;
    }
};

using SelfMultiHeadAttentionParams =
        std::tuple<Heads, SequenceLength, EmbeddingSize, IsCausal, AttentionMask, HasScale, Kind>;

class SelfMultiHeadAttentionLayerTest :
        public SdpaLayerTestBase<SelfMultiHeadAttentionLayerTest>,
        public ::testing::WithParamInterface<SelfMultiHeadAttentionParams> {
public:
    std::vector<InputShape> generateInputShapes(SelfMultiHeadAttentionParams params) {
        const auto& [heads, sequenceLength, embeddingSize, isCausal, hasAttentionMask, hasScale, kind] = params;

        auto qShape = generateTestShape(heads.value(), sequenceLength.value(), embeddingSize.value());
        auto kShape = generateTestShape(heads.value(), sequenceLength.value(), embeddingSize.value());
        auto vShape = generateTestShape(heads.value(), sequenceLength.value(), embeddingSize.value());

        auto inputShapes = std::vector<InputShape>{qShape, kShape, vShape};

        if (hasAttentionMask.value() == Mask::Full) {
            auto attentionMaskShape = generateTestShape(heads.value(), sequenceLength.value(), sequenceLength.value());
            inputShapes.push_back(attentionMaskShape);
        } else if (hasAttentionMask.value() == Mask::Broadcasted) {
            auto attentionMaskShape = generateTestShape(1, sequenceLength.value(), sequenceLength.value());
            inputShapes.push_back(attentionMaskShape);
        }

        return inputShapes;
    }
};

}  // namespace

TEST_P(CrossMultiHeadAttentionLayerTest, NPU5010_HW) {
    abs_threshold = 0.03f;
    run(Platform::NPU5010);
}

TEST_P(SelfMultiHeadAttentionLayerTest, NPU5010_HW) {
    abs_threshold = 0.03f;
    run(Platform::NPU5010);
}

//
// MultiHeadAttentionLayerTest
//

INSTANTIATE_TEST_SUITE_P(smoke, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                              //
                                            ::testing::Values(SourceSeqLen{32}),                      //
                                            ::testing::Values(TargetSeqLen{64}),                      //
                                            ::testing::Values(QKEmbeddingSize{64}),                   //
                                            ::testing::Values(VEmbeddingSize{128}),                   //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Full,
                                                                                           Mask::Absent}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),        //
                                            ::testing::ValuesIn(std::vector<Kind>{ALL_KINDS})),             //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke, SelfMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                              //
                                            ::testing::Values(SequenceLength{64}),                    //
                                            ::testing::Values(EmbeddingSize{256}),                    //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,
                                                                                           Mask::Absent}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),        //
                                            ::testing::ValuesIn(std::vector<Kind>{ALL_KINDS})),             //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_BigSequenceLength, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                                         //
                                            ::testing::ValuesIn(std::vector<SourceSeqLen>{128}),                 //
                                            ::testing::ValuesIn(std::vector<TargetSeqLen>{8 * 1024}),            //
                                            ::testing::Values(QKEmbeddingSize{32}),                              //
                                            ::testing::Values(VEmbeddingSize{64}),                               //
                                            ::testing::ValuesIn(std::vector<IsCausal>{false}),                   //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true}),                    //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),                      //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_CornerCases, CrossMultiHeadAttentionLayerTest,
                         ::testing::ValuesIn(std::vector<CrossMultiHeadAttentionParams>{
                                 CrossMultiHeadAttentionParams{Heads{25}, SourceSeqLen{1024}, TargetSeqLen{1},
                                                               QKEmbeddingSize{64}, VEmbeddingSize{64}, IsCausal{true},
                                                               AttentionMask{Mask::Absent}, HasScale{false},
                                                               Kind{SDPA::FlashAttention}},
                         }),
                         PrintTestCaseName());

//
// Real networks
//

INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_Phi_3_5_mini, SelfMultiHeadAttentionLayerTest,        //
                         ::testing::Combine(::testing::Values(Heads{32}),                         //
                                            ::testing::Values(SequenceLength{1024}),              //
                                            ::testing::Values(EmbeddingSize{96}),                 //
                                            ::testing::Values(IsCausal{false}),                   //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),  //
                                            ::testing::Values(HasScale{true}),                    //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),       //
                         PrintTestCaseName());

// The same shape with Llama-3.1-8B and Qwen3-4B
INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_Mistral_7B, SelfMultiHeadAttentionLayerTest,          //
                         ::testing::Combine(::testing::Values(Heads{32}),                         //
                                            ::testing::Values(SequenceLength{1024}),              //
                                            ::testing::Values(EmbeddingSize{128}),                //
                                            ::testing::Values(IsCausal{false}),                   //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),  //
                                            ::testing::Values(HasScale{true}),                    //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),       //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_Llama_2_13B, SelfMultiHeadAttentionLayerTest,         //
                         ::testing::Combine(::testing::Values(Heads{40}),                         //
                                            ::testing::Values(SequenceLength{1024}),              //
                                            ::testing::Values(EmbeddingSize{128}),                //
                                            ::testing::Values(IsCausal{false}),                   //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),  //
                                            ::testing::Values(HasScale{true}),                    //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),       //
                         PrintTestCaseName());

// Fails on CI E#180955
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_WLM_Reproducer, CrossMultiHeadAttentionLayerTest,         //
                         ::testing::Combine(::testing::Values(Heads{32}),                         //
                                            ::testing::Values(SourceSeqLen{1024 * 3}),            //
                                            ::testing::Values(TargetSeqLen{1024}),                //
                                            ::testing::Values(QKEmbeddingSize{128}),              //
                                            ::testing::Values(VEmbeddingSize{128}),               //
                                            ::testing::Values(IsCausal{false}),                   //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),  //
                                            ::testing::Values(HasScale{true}),                    //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),       //
                         PrintTestCaseName());

// Fails on CI E#191165
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Padding_Reproducer, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{16}),                                 //
                                            ::testing::ValuesIn(std::vector<SourceSeqLen>{249}),          //
                                            ::testing::ValuesIn(std::vector<TargetSeqLen>{117}),          //
                                            ::testing::ValuesIn(std::vector<QKEmbeddingSize>{85}),        //
                                            ::testing::ValuesIn(std::vector<VEmbeddingSize>{87}),         //
                                            ::testing::ValuesIn(std::vector<IsCausal>{false}),            //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Full}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true}),             //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),               //
                         PrintTestCaseName());

// Fails to compile E#191304
// Strides '[33408, 5568, 87, 1]' do not match with shape '[1, 3, 64, 96]' and order 'NCHW'
INSTANTIATE_TEST_SUITE_P(DISABLED_Padding_compilation_fail, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{16}),                                 //
                                            ::testing::ValuesIn(std::vector<SourceSeqLen>{256}),          //
                                            ::testing::ValuesIn(std::vector<TargetSeqLen>{64}),           //
                                            ::testing::ValuesIn(std::vector<QKEmbeddingSize>{96}),        //
                                            ::testing::ValuesIn(std::vector<VEmbeddingSize>{87}),         //
                                            ::testing::ValuesIn(std::vector<IsCausal>{false}),            //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Full}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true}),             //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),               //
                         PrintTestCaseName());

//
// Manual performance measurements
//

namespace {

template <typename Type, int N>
constexpr auto generateSeqLenRange() {
    std::array<Type, N> params{};
    for (int i = 0; i < N; ++i) {
        params[i] = Type{(i + 1) * 1024};
    }
    return params;
}

}  // namespace

// Intentionally disabled
INSTANTIATE_TEST_SUITE_P(DISABLED_Perf_OneKernel, SelfMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{32}),                         //
                                            ::testing::Values(SequenceLength{96}),                //
                                            ::testing::Values(EmbeddingSize{96}),                 //
                                            ::testing::Values(IsCausal{false}),                   //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),  //
                                            ::testing::Values(HasScale{true}),                    //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),       //
                         PrintTestCaseName());

// This SDPA shape is used in Mistral-7B, Llama-3.1-8B, Qwen3-4B
INSTANTIATE_TEST_SUITE_P(DISABLED_Perf_PrefillStageNPUW, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{32}),                                 //
                                            ::testing::ValuesIn(generateSeqLenRange<SourceSeqLen, 8>()),  //
                                            ::testing::Values(TargetSeqLen{1024}),                        //
                                            ::testing::Values(QKEmbeddingSize{128}),                      //
                                            ::testing::Values(VEmbeddingSize{128}),                       //
                                            ::testing::Values(IsCausal{false}),                           //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),          //
                                            ::testing::Values(HasScale{true}),                            //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),               //
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(DISABLED_Perf_GenerateStage, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{32}),                                 //
                                            ::testing::ValuesIn(generateSeqLenRange<SourceSeqLen, 8>()),  //
                                            ::testing::Values(TargetSeqLen{1}),                           //
                                            ::testing::Values(QKEmbeddingSize{128}),                      //
                                            ::testing::Values(VEmbeddingSize{128}),                       //
                                            ::testing::Values(IsCausal{false}),                           //
                                            ::testing::Values(AttentionMask{Mask::Broadcasted}),          //
                                            ::testing::Values(HasScale{true}),                            //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),               //
                         PrintTestCaseName());

//
// Dynamic CrossMultiHeadAttentionLayerTest
//

// [Tracking number: E#160081]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Dynamic, CrossMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{12}),                             //
                                            ::testing::Values(SourceSeqLen{512_Dyn}),                 //
                                            ::testing::Values(TargetSeqLen{512_Dyn}),                 //
                                            ::testing::Values(QKEmbeddingSize{64}),                   //
                                            ::testing::Values(VEmbeddingSize{64}),                    //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<AttentionMask>{Mask::Broadcasted,
                                                                                           Mask::Absent}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false}),        //
                                            ::testing::Values(Kind{SDPA::FlashAttention})),
                         PrintTestCaseName());

}  // namespace ov::test
