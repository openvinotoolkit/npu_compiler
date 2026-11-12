//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset14.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/pass/manager.hpp>
#include <pretty_test_arguments.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/env.hpp"

using namespace ov::test::utils;
using namespace ov::test;

enum class MaskType {
    NONE,     // No mask
    DEFAULT,  // Float mask
    CAUSAL    // Causal mask (generated internally)
};

struct SDPAParams {
    ov::Shape inputQ;
    ov::Shape inputK;
    ov::Shape inputV;
    ov::Shape inputMask;
    MaskType maskType = MaskType::DEFAULT;
    bool hasScale = true;
    bool hasSink = false;
    ov::Shape inputSink{};
};

class ScaledDotProductAttentionV14LayerTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SDPAParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto testParams = GetParam();
        const bool hasSink = testParams.hasSink;

        const size_t sinkIndex = hasSink ? funcInputs.size() - 1 : SIZE_MAX;
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = (i == sinkIndex) ? 4 : 1;
            in_data.resolution = 32768;
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }
    void SetUp() override {
        auto elementType = ov::element::f16;
        inType = outType = elementType;
        const auto testParams = GetParam();
        const MaskType maskType = testParams.maskType;
        const bool hasScale = testParams.hasScale;
        const bool hasSink = testParams.hasSink;
        const bool isCausal = (maskType == MaskType::CAUSAL);
        const bool hasAttentionMask = (maskType != MaskType::NONE && maskType != MaskType::CAUSAL);

        // For SDPA spec: when scale is present without mask, we need a dummy mask parameter
        const bool needsMaskParam = hasAttentionMask || (hasScale && !isCausal) || hasSink;

        const auto inputQShape = testParams.inputQ;
        const auto inputKShape = testParams.inputK;
        const auto inputVShape = testParams.inputV;
        const auto inputMaskShape = hasAttentionMask ? testParams.inputMask : ov::Shape{1, 1, 1, 1};

        // Build input shapes
        std::vector<ov::Shape> inputShapes = {inputQShape, inputKShape, inputVShape};
        if (needsMaskParam) {
            inputShapes.push_back(inputMaskShape);
        }
        if (hasSink) {
            inputShapes.push_back(testParams.inputSink);
        }
        init_input_shapes(ov::test::static_shapes_to_test_representation(inputShapes));

        // Create Q, K, V parameters
        const auto inputQ = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(0));
        inputQ->set_friendly_name("inputQ");
        const auto inputK = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(1));
        inputK->set_friendly_name("inputK");
        const auto inputV = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(2));
        inputV->set_friendly_name("inputV");

        // Create mask parameter based on mask type
        size_t nextIdx = 3;
        std::shared_ptr<ov::opset3::Parameter> inputMask;
        if (needsMaskParam) {
            inputMask = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(nextIdx));
            inputMask->set_friendly_name("inputMask");
            nextIdx++;
        }

        std::shared_ptr<ov::opset3::Parameter> inputSinkParam;
        if (hasSink) {
            inputSinkParam = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(nextIdx));
            inputSinkParam->set_friendly_name("inputSink");
        }

        // Build parameter vector
        ov::ParameterVector inputParams = {inputQ, inputK, inputV};
        if (needsMaskParam) {
            inputParams.push_back(inputMask);
        }
        if (hasSink) {
            inputParams.push_back(inputSinkParam);
        }

        // Build inputs for SDPA operation
        ov::OutputVector inputs;
        inputs.emplace_back(inputQ);
        inputs.emplace_back(inputK);
        inputs.emplace_back(inputV);
        if (needsMaskParam) {
            inputs.emplace_back(inputMask);
        }

        if (hasScale || hasSink) {
            const auto inputQRank = inputQShape.size();
            const auto scaleFactor = 1 / sqrt(inputQShape[inputQRank - 1]);
            const auto scale = ov::op::v0::Constant::create(elementType, ov::Shape{1}, {scaleFactor});
            scale->set_friendly_name("constantScale");
            inputs.emplace_back(scale);
        }

        if (hasSink) {
            inputs.emplace_back(inputSinkParam);
        }

        const auto sdpa = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCausal);
        sdpa->set_friendly_name("sdpa");

        auto results = ov::ResultVector();
        for (size_t i = 0; i < sdpa->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(sdpa->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "SDPA");
        functionRefs = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
    }

    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-decompose-sdpa=false";
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<SDPAParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        const auto& p = obj.param;
        result << "TestIdx=" << obj.index << sep;
        result << "Q=" << p.inputQ << sep;
        result << "K=" << p.inputK << sep;
        result << "V=" << p.inputV << sep;

        switch (p.maskType) {
        case MaskType::NONE:
            result << "Mask=NONE" << sep;
            break;
        case MaskType::DEFAULT:
            result << "Mask=DEFAULT_" << p.inputMask << sep;
            break;
        case MaskType::CAUSAL:
            result << "Mask=CAUSAL" << sep;
            break;
        }

        p.hasScale ? result << "HasScale=true" : result << "HasScale=false";
        if (p.hasSink) {
            result << sep << "Sink=" << p.inputSink;
        }
        return result.str();
    };
};

class ScaledDotProductAttentionV14SinkLayerTestCommon : public ScaledDotProductAttentionV14LayerTestCommon {};

TEST_P(ScaledDotProductAttentionV14LayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ScaledDotProductAttentionV14SinkLayerTestCommon, NPU5010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ScaledDotProductAttentionV14SinkLayerTestCommon, NPU6010_HW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

INSTANTIATE_TEST_SUITE_P(precommit, ScaledDotProductAttentionV14LayerTestCommon,
                         ::testing::ValuesIn({
                                 // SelfAttention Tests
                                 SDPAParams{{1, 1, 1, 8}, {1, 1, 16, 8}, {1, 1, 16, 8}, {1, 1, 1, 16}},
                                 SDPAParams{{1, 1, 12, 8}, {1, 1, 16, 8}, {1, 1, 16, 8}, {1, 1, 12, 16}},
                                 SDPAParams{{1, 32, 12, 8}, {1, 32, 16, 8}, {1, 32, 16, 8}, {1, 32, 12, 16}},

                                 // CrossAttention Tests
                                 SDPAParams{{1, 1, 1, 8}, {1, 1, 16, 8}, {1, 1, 16, 4}, {1, 1, 1, 16}},
                                 SDPAParams{{1, 1, 12, 8}, {1, 1, 16, 8}, {1, 1, 16, 4}, {1, 1, 12, 16}},
                                 SDPAParams{{1, 8, 12, 8}, {1, 8, 16, 8}, {1, 8, 16, 4}, {1, 8, 12, 16}},
                         }),
                         ScaledDotProductAttentionV14LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_TestOptionalInputs, ScaledDotProductAttentionV14LayerTestCommon,
        ::testing::ValuesIn({
                // hasAttentionMask = true, hasScale = true
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 18, 12, 16}, MaskType::DEFAULT, true},

                // hasAttentionMask = true, hasScale = false
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 18, 12, 16}, MaskType::DEFAULT, false},

                // hasAttentionMask = false (NONE), hasScale = true
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 1, 1, 1}, MaskType::NONE, true},

                // hasAttentionMask = false (NONE), hasScale = false
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 1, 1, 1}, MaskType::NONE, false},
        }),
        ScaledDotProductAttentionV14LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_NetworksConfigurations, ScaledDotProductAttentionV14LayerTestCommon,
                         ::testing::ValuesIn({
                                 // Phi-3-mini configuration
                                 SDPAParams{{1, 32, 1, 96}, {1, 32, 1024, 96}, {1, 32, 1024, 96}, {1, 1, 1, 1024}},

                                 // Llama-2-7b configuration
                                 SDPAParams{{1, 32, 1, 128}, {1, 32, 1024, 128}, {1, 32, 1024, 128}, {1, 1, 1, 1024}},

                                 // Transformer_complex
                                 SDPAParams{{1, 1, 55, 128}, {1, 1, 49, 128}, {1, 1, 49, 128}, {1, 1, 1, 49}},
                                 SDPAParams{{1, 1, 55, 128}, {1, 1, 55, 128}, {1, 1, 55, 128}, {1, 1, 1, 55}},
                                 SDPAParams{{1, 1, 49, 128}, {1, 1, 49, 128}, {1, 1, 49, 128}, {1, 1, 1, 49}},
                                 SDPAParams{{1, 1, 55, 128}, {1, 1, 55, 128}, {1, 1, 55, 128}, {1, 1, 55, 55}},

                                 // miniCPM
                                 SDPAParams{{1, 24, 1, 64}, {1, 24, 1024, 64}, {1, 24, 1024, 64}, {1, 1, 1, 1024}},

                                 // Other configurations
                                 SDPAParams{{1, 1, 1, 64}, {1, 1, 64, 64}, {1, 1, 64, 64}, {1, 1, 1, 64}},
                                 SDPAParams{{1, 1, 64, 64}, {1, 1, 64, 64}, {1, 1, 64, 64}, {1, 1, 64, 64}},
                                 SDPAParams{{1, 8, 25, 64}, {1, 8, 475, 64}, {1, 8, 475, 64}, {1, 8, 25, 475}},
                                 SDPAParams{{1, 12, 77, 96}, {1, 12, 77, 96}, {1, 12, 77, 64}, {1, 1, 1, 77}},
                         }),
                         ScaledDotProductAttentionV14LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_AttentionWithSinkSupport, ScaledDotProductAttentionV14SinkLayerTestCommon,
        ::testing::ValuesIn({
                // Model-like sink pattern: [B,H,tSL,E] x [B,H,E,S], mask [B,H,tSL,S], sink [B,H,tSL,1].
                // Mask and Sink can be broadcasted on B and H dimensions. Mask on all dimensions.
                SDPAParams{{1, 64, 1024, 64},
                           {1, 64, 1024, 64},
                           {1, 64, 1024, 64},
                           {1, 1, 1024, 1024},
                           MaskType::DEFAULT,
                           true,
                           true,
                           {1, 64, 1024, 1}},
                SDPAParams{{1, 64, 1024, 64},
                           {1, 64, 1024, 64},
                           {1, 64, 1024, 64},
                           {1, 1, 1024, 1024},
                           MaskType::DEFAULT,
                           true,
                           true,
                           {1, 64, 1, 1}},
        }),
        ScaledDotProductAttentionV14SinkLayerTestCommon::getTestCaseName);
