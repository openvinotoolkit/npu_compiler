// Copyright (C) 2025 Intel Corporation
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

struct SDPAParams {
    ov::Shape inputQ;
    ov::Shape inputK;
    ov::Shape inputV;
    ov::Shape inputMask;
    bool hasAttentionMask = true;
    bool hasScale = true;
    bool isCausal = false;
};

class ScaledDotProductAttentionV14LayerTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SDPAParams> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = 0;
            in_data.range = 1;
            in_data.resolution = 32768;
            ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                            targetInputStaticShapes[i], in_data);
            VpuOv2LayerTest::inputs.insert({funcInputs[i].get_node_shared_ptr(), tensorData});
        }
    }
    void SetUp() override {
        auto elementType = ov::element::f32;
        inType = outType = elementType;
        const auto testParams = GetParam();
        const bool hasAttentionMask = testParams.hasAttentionMask;
        const bool hasScale = testParams.hasScale;
        const bool isCausal = testParams.isCausal;

        const auto inputQShape = testParams.inputQ;
        const auto inputKShape = testParams.inputK;
        const auto inputVShape = testParams.inputV;
        const auto inputMaskShape = !hasAttentionMask && hasScale ? ov::Shape{1, 1, 1, 1} : testParams.inputMask;

        init_input_shapes(ov::test::static_shapes_to_test_representation(
                {inputQShape, inputKShape, inputVShape, inputMaskShape}));

        const auto inputQ = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(0));
        const auto inputK = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(1));
        const auto inputV = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(2));
        const auto inputMask = std::make_shared<ov::opset3::Parameter>(inType, inputDynamicShapes.at(3));

        ov::ParameterVector inputParams;
        inputParams.push_back(inputQ);
        inputParams.push_back(inputK);
        inputParams.push_back(inputV);
        if (hasAttentionMask || (hasScale && !hasAttentionMask)) {
            inputParams.push_back(inputMask);
        }

        ov::OutputVector inputs;
        for (auto& input : inputParams) {
            inputs.emplace_back(input);
        }

        if (hasScale) {
            const auto inputQRank = inputQShape.size();
            const auto scaleFactor = 1 / sqrt(inputQShape[inputQRank - 1]);
            const auto scaleShape = ov::Shape{1};
            const auto scale = ov::op::v0::Constant::create(elementType, scaleShape, {scaleFactor});
            inputs.emplace_back(scale);
        }

        const auto sdp = std::make_shared<ov::opset14::ScaledDotProductAttention>(inputs, isCausal);
        sdp->set_friendly_name("sdp");

        auto results = ov::ResultVector();
        for (size_t i = 0; i < sdp->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(sdp->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "SDP");
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
        p.hasAttentionMask ? result << "Mask=" << p.inputMask << sep : result << "Mask=none" << sep;
        p.hasScale ? result << "HasScale=true" << sep : result << "HasScale=false" << sep;
        p.isCausal ? result << "Causal=true" : result << "Causal=false";
        return result.str();
    };
};

TEST_P(ScaledDotProductAttentionV14LayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.012;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
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
                // hasAttentionMask = true, hasScale = true, isCausal = false
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 18, 12, 16}, true, true, false},

                // hasAttentionMask = true, hasScale = false, isCausal = false
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 18, 12, 16}, true, false, false},

                // hasAttentionMask = false, hasScale = true, isCausal = false
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 18, 12, 16}, false, true, false},

                // hasAttentionMask = false, hasScale = false, isCausal = false
                SDPAParams{{1, 18, 12, 8}, {1, 18, 16, 8}, {1, 18, 16, 4}, {1, 18, 12, 16}, false, false, false},
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
