//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv_dtypes_test.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace test {

using namespace ov::test::utils;

namespace {
constexpr float K_ABS_THRESHOLD = 0.5f;
constexpr float K_REL_THRESHOLD = 0.05f;
constexpr float K_PER_AXIS_SCALE_VARIATION = 0.01f;
constexpr size_t K_PER_AXIS_VARIATION_PERIOD = 5;
constexpr float K_PER_AXIS_CENTER = 2.0f;
}  // namespace

void ConvDtypesTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::test::utils::InputGenerateData inGenData;
        inGenData.start_from = -1.f;
        inGenData.range = 2.f;
        inGenData.resolution = 256;
        inGenData.seed = 1;
        auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                                              inGenData);
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

std::string ConvDtypesTest::getTestCaseName(const testing::TestParamInfo<convDtypesTestParamsSet>& obj) {
    const auto& [convParams, modelType, inputShape] = obj.param;
    const auto& [filterStorageType, inputQuantizationLevels, outputStorageType, outputQuantizationLevels,
                 perAxisWeights] = convParams;

    const std::string sep = "_";
    std::ostringstream result;

    result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
    result << "TestIdx=" << obj.index << sep;
    result << "InputShape=" << vec2str(inputShape) << sep;
    result << "FilterStorage=" << filterStorageType << sep;
    result << "InputQuantLevels=" << inputQuantizationLevels << sep;
    result << "OutputStorage=" << outputStorageType << sep;
    result << "OutputQuantLevels=" << outputQuantizationLevels << sep;
    result << "PerAxis=" << (perAxisWeights ? "true" : "false") << sep;
    result << "netPRC=" << modelType << sep;

    return result.str();
}

void ConvDtypesTest::SetUp() {
    abs_threshold = K_ABS_THRESHOLD;
    rel_threshold = K_REL_THRESHOLD;

    const auto& [convParams, modelType, inputShape] = this->GetParam();
    const auto filterStorageType = std::get<0>(convParams);
    const auto inputQuantizationLevels = std::get<1>(convParams);
    const auto outputStorageType = std::get<2>(convParams);
    const auto outputQuantizationLevels = std::get<3>(convParams);
    const auto perAxisWeights = std::get<4>(convParams);

    init_input_shapes(static_shapes_to_test_representation({inputShape}));

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front())};

    const auto fqNode =
            make_fake_quantize(params[0], modelType, inputQuantizationLevels, {}, {-1.f}, {1.f}, {0.f}, {1.f});

    static constexpr size_t convOutChannels = 64;
    const size_t inputChannels = inputShape[1];
    const std::vector<size_t> weightsShape{convOutChannels, inputChannels, 3, 3};

    const auto weightsConst = make_constant(filterStorageType, weightsShape, ov::test::utils::InputGenerateData(0, 20));
    const auto weightsFP16 = std::make_shared<ov::op::v0::Convert>(weightsConst, modelType);

    const auto weightMaxValue = static_cast<float>((1u << filterStorageType.bitwidth()) - 1);

    std::shared_ptr<ov::Node> convWeights;
    if (perAxisWeights) {
        std::vector<float> perChannelScales;
        perChannelScales.reserve(convOutChannels);

        for (size_t i = 0; i < convOutChannels; ++i) {
            const float scale =
                    (1.0f / weightMaxValue) *
                    (1.0f + K_PER_AXIS_SCALE_VARIATION *
                                    (static_cast<float>(i % K_PER_AXIS_VARIATION_PERIOD) - K_PER_AXIS_CENTER));
            perChannelScales.push_back(scale);
        }
        const auto zScaleConst =
                ov::op::v0::Constant::create(modelType, ov::Shape{convOutChannels, 1, 1, 1}, perChannelScales);
        convWeights = std::make_shared<ov::op::v1::Multiply>(weightsFP16, zScaleConst);
    } else {
        const float zScale = 1.0f / weightMaxValue;
        const auto zScaleConst =
                ov::op::v0::Constant::create(modelType, ov::Shape{1, 1, 1, 1}, std::vector<float>{zScale});
        convWeights = std::make_shared<ov::op::v1::Multiply>(weightsFP16, zScaleConst);
    }

    const auto conv = std::make_shared<ov::op::v1::Convolution>(fqNode, convWeights, ov::Strides{1, 1},
                                                                ov::CoordinateDiff{1, 1}, ov::CoordinateDiff{1, 1},
                                                                ov::Strides{1, 1}, ov::op::PadType::EXPLICIT);

    std::shared_ptr<ov::Node> modelOutput = conv;
    if (outputStorageType != ov::element::f16) {
        modelOutput = make_fake_quantize(conv, modelType, outputQuantizationLevels, {}, {0.f}, {100.f}, {0.f}, {100.f});
    }

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(modelOutput)};
    function = std::make_shared<ov::Model>(results, params, "ConvDtypes");
}

}  // namespace test
}  // namespace ov
