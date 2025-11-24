//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/node_vector.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/opsets/opset14_decl.hpp>
#include <openvino/pass/manager.hpp>
#include <string_view>

#include <common/print_test_case_name.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <pretty_test_arguments.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "private_properties.hpp"
#include "vpux/utils/core/env.hpp"

namespace ov::test {

// Batch == 1
PRETTY_PARAM(Heads, BoundedDim);            // H
PRETTY_PARAM(SourceSeqLen, BoundedDim);     // S
PRETTY_PARAM(TargetSeqLen, BoundedDim);     // L
PRETTY_PARAM(QKEmbeddingSize, BoundedDim);  // E
PRETTY_PARAM(VEmbeddingSize, BoundedDim);   // Ev
PRETTY_PARAM(IsCausal, bool);
PRETTY_PARAM(HasAttentionMask, bool);
PRETTY_PARAM(HasScale, bool);

namespace {

ov::ParameterVector generateInputParams(const std::vector<ov::PartialShape>& inputDynamicShapes,
                                        HasAttentionMask hasAttentionMask, HasScale hasScale) {
    ov::ParameterVector inputParams;

    const auto inputType = ov::element::f32;
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[0]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[1]));
    inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[2]));

    inputParams[0]->set_friendly_name("query");
    inputParams[1]->set_friendly_name("key");
    inputParams[2]->set_friendly_name("value");

    if (hasScale) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(
                inputType, hasAttentionMask ? inputDynamicShapes[3] : ov::PartialShape{}));
        inputParams.back()->set_friendly_name("attention_mask");

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, ov::PartialShape{1}));
        inputParams.back()->set_friendly_name("scale");
    } else if (hasAttentionMask) {
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, inputDynamicShapes[3]));
        inputParams.back()->set_friendly_name("attention_mask");
    }

    return inputParams;
}

}  // namespace

template <typename Derived>
class SdpaLayerTestBase : public VpuOv2LayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        auto shapes = std::vector<ov::Shape>{targetInputStaticShapes[0], targetInputStaticShapes[1],
                                             targetInputStaticShapes[2]};
        if (hasScale) {
            shapes.push_back(hasAttentionMask ? targetInputStaticShapes[3] : ov::Shape{});
            shapes.push_back(ov::Shape{1});  // NOLINT emplace_back leads to incorrect results
        } else if (hasAttentionMask) {
            shapes.push_back(targetInputStaticShapes[3]);
        }

        SubgraphBaseTest::generate_inputs(shapes);
    }

    void SetUp() override {
        auto params = static_cast<Derived*>(this)->GetParam();

        this->hasAttentionMask = std::get<HasAttentionMask>(params);
        this->hasScale = std::get<HasScale>(params);

        const auto inputShapes = static_cast<Derived*>(this)->generateInputShapes(params);

        init_input_shapes(inputShapes);

        const auto inputParams = generateInputParams(inputDynamicShapes, hasAttentionMask, hasScale);

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        auto sdpa = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, std::get<IsCausal>(params));
        sdpa->set_friendly_name("sdpa");

        function = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, inputParams, "SDPA");
    }

public:
    bool hasAttentionMask = false;
    bool hasScale = false;
};

//
// MultiHeadAttentionParams
//

using MultiHeadAttentionParams = std::tuple<Heads, SourceSeqLen, TargetSeqLen, QKEmbeddingSize, VEmbeddingSize,
                                            IsCausal, HasAttentionMask, HasScale>;

class MultiHeadAttentionLayerTest :
        public testing::WithParamInterface<MultiHeadAttentionParams>,
        public SdpaLayerTestBase<MultiHeadAttentionLayerTest> {
public:
    std::vector<InputShape> generateInputShapes(MultiHeadAttentionParams params) {
        const auto& [heads, sourceSeqLen, targetSeqLen, qkEmbeddingSize, vEmbeddingSize, isCausal, hasAttentionMask,
                     hasScale] = params;

        auto qShape = generateTestShape(heads, targetSeqLen, qkEmbeddingSize);
        auto kShape = generateTestShape(heads, sourceSeqLen, qkEmbeddingSize);
        auto vShape = generateTestShape(heads, sourceSeqLen, vEmbeddingSize);

        auto inputShapes = std::vector<InputShape>{qShape, kShape, vShape};

        if (hasAttentionMask) {
            auto attentionMaskShape = generateTestShape(heads, targetSeqLen, sourceSeqLen);
            inputShapes.push_back(attentionMaskShape);
        }

        return inputShapes;
    }
};

//
// FlashMultiHeadAttentionLayerTest
//

class FlashMultiHeadAttentionLayerTest : public MultiHeadAttentionLayerTest {
public:
    void SetUp() override {
        MultiHeadAttentionLayerTest::SetUp();

        // Disable OpenVINO ScaledDotProductAttentionDecomposition pass
        vpux::env::setEnvVar("NPU_DECOMPOSE_SDPA", "0");
    }

    void TearDown() override {
        MultiHeadAttentionLayerTest::TearDown();

        vpux::env::unsetEnvVar("NPU_DECOMPOSE_SDPA");
    }

    void configure_model() override {
        // Enable a pass that performs SDPA conversion to FlashSDPA
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-flash-sdpa-conversion=true";

#if !defined(_WIN32)
        setSkipInferenceCallback([](std::stringstream& ss) {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            const auto suite = std::string_view{info->test_suite_name()};
            const auto testName = std::string_view{info->name()};

            if (suite.find("smoke_CornerCases") != std::string_view::npos &&
                testName.find("NPU3720") != std::string_view::npos) {
                ss << "Skip inference for CornerCases on NPU3720/Ubuntu E#187570";
            }
        });
#endif
    }
};

TEST_P(MultiHeadAttentionLayerTest, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FlashMultiHeadAttentionLayerTest, NPU3720_HW) {
#if defined(_WIN32)
    abs_threshold = 0.02;
#else
    abs_threshold = 0.01;
#endif
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MultiHeadAttentionLayerTest, NPU4000_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(FlashMultiHeadAttentionLayerTest, NPU4000_HW) {
#if defined(_WIN32)
    abs_threshold = 0.03;
#else
    abs_threshold = 0.01;
#endif
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

//
// MultiHeadAttentionLayerTest
//

INSTANTIATE_TEST_SUITE_P(smoke, MultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                                      //
                                            ::testing::Values(SourceSeqLen{32}),                              //
                                            ::testing::Values(TargetSeqLen{64}),                              //
                                            ::testing::Values(QKEmbeddingSize{64}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                           //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),          //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false})           //
                                            ),
                         PrintTestCaseName());

//
// FlashMultiHeadAttentionLayerTest
//

INSTANTIATE_TEST_SUITE_P(smoke, FlashMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                                      //
                                            ::testing::Values(SourceSeqLen{32}),                              //
                                            ::testing::Values(TargetSeqLen{64}),                              //
                                            ::testing::Values(QKEmbeddingSize{64}),                           //
                                            ::testing::Values(VEmbeddingSize{128}),                           //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),          //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false})           //
                                            ),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_BigSequenceLength, FlashMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{8}),                               //
                                            ::testing::ValuesIn(std::vector<SourceSeqLen>{128}),       //
                                            ::testing::ValuesIn(std::vector<TargetSeqLen>{8 * 1024}),  //
                                            ::testing::Values(QKEmbeddingSize{32}),                    //
                                            ::testing::Values(VEmbeddingSize{64}),                     //
                                            ::testing::ValuesIn(std::vector<IsCausal>{false}),         //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true})           //
                                            ),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_CornerCases, FlashMultiHeadAttentionLayerTest,
                         ::testing::ValuesIn(std::vector<MultiHeadAttentionParams>{
                                 MultiHeadAttentionParams{Heads{25}, SourceSeqLen{1024}, TargetSeqLen{1},
                                                          QKEmbeddingSize{64}, VEmbeddingSize{64}, IsCausal{true},
                                                          HasAttentionMask{false}, HasScale{false}},
                         }),
                         PrintTestCaseName());

//
// Real networks
//

INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_Phi_3_mini, FlashMultiHeadAttentionLayerTest,
                         ::testing::Values(MultiHeadAttentionParams{
                                 Heads{32}, SourceSeqLen{1024}, TargetSeqLen{1024}, QKEmbeddingSize{96},
                                 VEmbeddingSize{96}, IsCausal{false}, HasAttentionMask{true}, HasScale{false}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_Llama_2_7b, FlashMultiHeadAttentionLayerTest,
                         ::testing::Values(MultiHeadAttentionParams{
                                 Heads{32}, SourceSeqLen{1024}, TargetSeqLen{1}, QKEmbeddingSize{128},
                                 VEmbeddingSize{128}, IsCausal{false}, HasAttentionMask{true}, HasScale{false}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_Transformer_complex, FlashMultiHeadAttentionLayerTest,
                         ::testing::Values(MultiHeadAttentionParams{
                                 Heads{1}, SourceSeqLen{49}, TargetSeqLen{55}, QKEmbeddingSize{128},
                                 VEmbeddingSize{128}, IsCausal{true}, HasAttentionMask{false}, HasScale{false}}),
                         PrintTestCaseName());

INSTANTIATE_TEST_SUITE_P(smoke_RealNetworks_miniCPM, FlashMultiHeadAttentionLayerTest,
                         ::testing::Values(MultiHeadAttentionParams{
                                 Heads{40}, SourceSeqLen{1024}, TargetSeqLen{1}, QKEmbeddingSize{96},
                                 VEmbeddingSize{96}, IsCausal{false}, HasAttentionMask{true}, HasScale{false}}),
                         PrintTestCaseName());

//
// Dynamic FlashMultiHeadAttentionLayerTest
//

// [Tracking number: E#160081]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Dynamic, FlashMultiHeadAttentionLayerTest,
                         ::testing::Combine(::testing::Values(Heads{12}),                                     //
                                            ::testing::Values(SourceSeqLen{512_Dyn}),                         //
                                            ::testing::Values(TargetSeqLen{512_Dyn}),                         //
                                            ::testing::Values(QKEmbeddingSize{64}),                           //
                                            ::testing::Values(VEmbeddingSize{64}),                            //
                                            ::testing::ValuesIn(std::vector<IsCausal>{true, false}),          //
                                            ::testing::ValuesIn(std::vector<HasAttentionMask>{true, false}),  //
                                            ::testing::ValuesIn(std::vector<HasScale>{true, false})           //
                                            ),
                         PrintTestCaseName());

}  // namespace ov::test
