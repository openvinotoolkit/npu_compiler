//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset1.hpp>
#include "openvino/opsets/opset6_decl.hpp"

#include "pretty_test_arguments.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using SkipConnectionSubgraphParams = std::tuple<ov::test::InputShape,  // Input shape
                                                ov::element::Type>;    // Input precision

class SkipConnectionSubgraph :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SkipConnectionSubgraphParams> {
public:
    void configure_model() override {
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SkipConnectionSubgraphParams>& obj) {
        const auto& [inShape, inType] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputShape={" << inShape.first.to_string() << "}" << sep;
        result << "InputType=" << inType;
        return result.str();
    }

    void SetUp() override {
        // Network schema:
        //
        //                   Input [1, 3, H, W]
        //                         |
        //                         |
        //                         v
        //                  Conv (stride 2x2)
        //                  3x3 kernel, 3→32 channels
        //                         |
        //                         | [1, 32, H/2, W/2]
        //                         |
        //         +---------------+---------------+
        //         |                               |
        //         | (skip connection)             |
        // NOTE:   | <-- SliceOp will be           v
        // This    |     inserted here        Conv2 (stride 1x1)
        // tests   |     by compiler!         3x3 kernel, 32→32 channels
        // SliceOp |                               |
        //         |                               | [1, 32, H/2, W/2]
        //         |                               |
        //         +-------------> Add <-----------+
        //                         |
        //                         |
        //                         v
        //                 Output [1, 32, H/2, W/2]
        //
        const auto& [inShape, inType] = GetParam();
        const int64_t inputChannels = 3;
        const int64_t expandedChannels = 32;
        const float weightValue = 1.f;

        init_input_shapes({inShape});

        auto input = std::make_shared<ov::opset1::Parameter>(inType, inputDynamicShapes[0]);

        std::vector<float> strideConvWeightsValues(expandedChannels * inputChannels * 3 * 3, weightValue);
        auto strideConvWeights = ov::opset1::Constant::create(ov::element::f32, {expandedChannels, inputChannels, 3, 3},
                                                              strideConvWeightsValues);

        const ov::Strides stride2x2 = {2, 2};
        const ov::CoordinateDiff padsBegin = {1, 1};
        const ov::CoordinateDiff padsEnd = {1, 1};
        const ov::Strides dilations = {1, 1};
        auto conv = std::make_shared<ov::opset6::Convolution>(input, strideConvWeights, stride2x2, padsBegin, padsEnd,
                                                              dilations);
        std::vector<float> convWeightsValues(expandedChannels * expandedChannels * 3 * 3, weightValue);
        auto strideConvWeights2 = ov::opset1::Constant::create(
                ov::element::f32, {expandedChannels, expandedChannels, 3, 3}, convWeightsValues);
        const ov::Strides strides1x1 = {1, 1};
        auto conv2 = std::make_shared<ov::opset6::Convolution>(conv, strideConvWeights2, strides1x1, padsBegin, padsEnd,
                                                               dilations);

        auto output = std::make_shared<ov::opset6::Add>(conv, conv2);

        const auto result = std::make_shared<ov::opset6::Result>(output);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input},
                                               "SkipConnectionSubgraph");
    }
};

class SkipConnectionSubgraphHostCompile : public SkipConnectionSubgraph {};

TEST_P(SkipConnectionSubgraphHostCompile, NPU5010_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU5010);
}

TEST_P(SkipConnectionSubgraphHostCompile, NPU4000_HC) {
    setHostCompileMode();
    setPluginCompilerType();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_SkipConnectionSubgraph_Dynamic_HC, SkipConnectionSubgraphHostCompile,
                         ::testing::Values(SkipConnectionSubgraphParams{
                                 /* inputShape = */ generateTestShape(std::vector<BoundedDim>{1, 3, 2056_Dyn, 2056_Dyn},
                                                                      hostCompileSmallShapesLimitationCallback),
                                 /* inType = */ ov::element::f32}),
                         SkipConnectionSubgraphHostCompile::getTestCaseName);

}  // namespace
