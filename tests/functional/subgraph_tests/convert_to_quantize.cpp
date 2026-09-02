//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/opsets/opset8.hpp"

namespace ov::test {

enum class PatternType { Simple, WithGroupConv };

struct QuantizationConfig {
    ov::element::Type outputType;
    double clampMin;
    double clampMax;
    double groupConvScale;  // For WithGroupConv pattern
    double groupConvBias;   // For WithGroupConv pattern (as Add layer)

    std::string toString() const {
        std::ostringstream oss;
        oss << outputType;
        return oss.str();
    }
};

class ConvertToQuantizeTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ov::Shape, ov::element::Type, PatternType, QuantizationConfig>> {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        // Generate input data based on the quantization range
        QuantizationConfig qConfig;
        ov::Shape inputShape;
        ov::element::Type inputType;
        PatternType patternType;
        std::tie(inputShape, inputType, patternType, qConfig) = GetParam();

        // Adjust input range based on clamp range
        const float rangeExtension = 5.0f;  // Generate values slightly outside clamp range
        float outputMin = static_cast<float>(qConfig.clampMin - rangeExtension);
        float outputMax = static_cast<float>(qConfig.clampMax + rangeExtension);

        // If GroupConv is used, we need to account for the transformation: output = input * scale + bias
        // To get the desired output range, we work backwards: input = (output - bias) / scale
        float startFrom, endAt;
        if (patternType == PatternType::WithGroupConv) {
            startFrom = (outputMin - static_cast<float>(qConfig.groupConvBias)) /
                        static_cast<float>(qConfig.groupConvScale);
            endAt = (outputMax - static_cast<float>(qConfig.groupConvBias)) /
                    static_cast<float>(qConfig.groupConvScale);
        } else {
            startFrom = outputMin;
            endAt = outputMax;
        }

        const uint32_t range = static_cast<uint32_t>(endAt - startFrom);
        const int32_t resolution = 100;

        ov::Tensor inputTensor = ov::test::utils::create_and_fill_tensor(
                funcInputs[0].get_element_type(), targetInputStaticShapes[0], range, startFrom, resolution);

        inputs.insert({funcInputs[0].get_node_shared_ptr(), inputTensor});
    }

    void SetUp() override {
        ov::Shape inputShape;
        ov::element::Type inputType;
        PatternType patternType;
        QuantizationConfig qConfig;
        std::tie(inputShape, inputType, patternType, qConfig) = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::opset8::Parameter>(inputType, inputDynamicShapes[0])};

        std::shared_ptr<ov::Node> patternInput = params[0];

        // Create GroupConvolution with Add if requested
        if (patternType == PatternType::WithGroupConv) {
            const auto channels = inputShape[1];
            std::vector<float> filterValues(channels, static_cast<float>(qConfig.groupConvScale));
            auto filterConst = ov::opset8::Constant::create(inputType, ov::Shape{channels, 1, 1, 1, 1}, filterValues);

            const ov::Strides strides = {1, 1};
            const ov::CoordinateDiff pads_begin = {0, 0};
            const ov::CoordinateDiff pads_end = {0, 0};
            const ov::Strides dilations = {1, 1};

            auto groupConv = std::make_shared<ov::opset8::GroupConvolution>(params[0], filterConst, strides, pads_begin,
                                                                            pads_end, dilations);

            std::vector<float> biasValues(channels, static_cast<float>(qConfig.groupConvBias));
            auto biasConst = ov::opset8::Constant::create(inputType, ov::Shape{1, channels, 1, 1}, biasValues);
            auto addOp = std::make_shared<ov::opset8::Add>(groupConv, biasConst);

            patternInput = addOp;
        }

        // Create the pattern: Round -> Clamp -> Convert
        const auto roundOp =
                std::make_shared<ov::opset8::Round>(patternInput, ov::opset8::Round::RoundMode::HALF_TO_EVEN);

        const auto clampOp = std::make_shared<ov::opset8::Clamp>(roundOp, qConfig.clampMin, qConfig.clampMax);

        const auto convertOp = std::make_shared<ov::opset8::Convert>(clampOp, qConfig.outputType);

        const ov::ResultVector results{std::make_shared<ov::opset8::Result>(convertOp)};

        const std::string testName = "ConvertToQuantize_" + qConfig.toString() +
                                     (patternType == PatternType::WithGroupConv ? "_WithGroupConv" : "_Simple");
        function = std::make_shared<ov::Model>(results, params, testName);
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<ov::Shape, ov::element::Type, PatternType, QuantizationConfig>>&
                    obj) {
        ov::Shape inputShape;
        ov::element::Type inputType;
        PatternType patternType;
        QuantizationConfig qConfig;
        std::tie(inputShape, inputType, patternType, qConfig) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "IS=" << ov::test::utils::vec2str(inputShape) << sep;
        result << "IT=" << inputType << sep;
        result << "OT=" << qConfig.outputType << sep;
        result << "Pattern=" << (patternType == PatternType::WithGroupConv ? "WithGroupConv" : "Simple") << sep;
        return result.str();
    }
};

class ConvertToQuantizeTest_NPU3720 : public ConvertToQuantizeTestCommon {};
class ConvertToQuantizeTest_NPU4000 : public ConvertToQuantizeTestCommon {};

TEST_P(ConvertToQuantizeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvertToQuantizeTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

class ConvertToQuantizeTest_NPU5010 : public ConvertToQuantizeTestCommon {};

TEST_P(ConvertToQuantizeTest_NPU5010, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

class ConvertToQuantizeTest_NPU5020 : public ConvertToQuantizeTestCommon {};

TEST_P(ConvertToQuantizeTest_NPU5020, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

namespace {

const std::vector<ov::Shape> inputShapes = {
        {1, 32, 32, 50},
};

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f16,
};

const std::vector<PatternType> patternTypes = {
        PatternType::Simple,
        PatternType::WithGroupConv,
};

// Quantization configurations for different data types
const std::vector<QuantizationConfig> quantizationConfigs = {
        // i8 (signed 8-bit): range [-128, 127]
        {ov::element::i8, -128.0, 127.0, 2.0, 0.0},

        // u8 (unsigned 8-bit): range [0, 255]
        {ov::element::u8, 0.0, 255.0, 2.0, 128.0},
};

// 4-bit quantization configs - only supported on NPU >= 4 (asm_quant_uniform_f16_u4/i4)
const std::vector<QuantizationConfig> quantizationConfigs4bit = {
        // u4 (unsigned 4-bit): range [0, 15]
        {ov::element::u4, 0.0, 15.0, 1.0, 0.0},

        // i4 (signed 4-bit): range [-8, 7]
        {ov::element::i4, -8.0, 7.0, 1.0, 0.0},
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize, ConvertToQuantizeTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(patternTypes),
                                            ::testing::ValuesIn(quantizationConfigs)),
                         ConvertToQuantizeTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize, ConvertToQuantizeTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(patternTypes),
                                            ::testing::ValuesIn(quantizationConfigs)),
                         ConvertToQuantizeTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize4bit, ConvertToQuantizeTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(PatternType::Simple),
                                            ::testing::ValuesIn(quantizationConfigs4bit)),
                         ConvertToQuantizeTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize, ConvertToQuantizeTest_NPU5010,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(patternTypes),
                                            ::testing::ValuesIn(quantizationConfigs)),
                         ConvertToQuantizeTest_NPU5010::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize4bit, ConvertToQuantizeTest_NPU5010,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(PatternType::Simple),
                                            ::testing::ValuesIn(quantizationConfigs4bit)),
                         ConvertToQuantizeTest_NPU5010::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize, ConvertToQuantizeTest_NPU5020,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(patternTypes),
                                            ::testing::ValuesIn(quantizationConfigs)),
                         ConvertToQuantizeTest_NPU5020::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToQuantize4bit, ConvertToQuantizeTest_NPU5020,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(PatternType::Simple),
                                            ::testing::ValuesIn(quantizationConfigs4bit)),
                         ConvertToQuantizeTest_NPU5020::getTestCaseName);

}  // namespace

}  // namespace ov::test
