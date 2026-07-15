//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6_decl.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"

using namespace ov::test::utils;
using namespace ov::test;

enum class ConvType { Regular, Depthwise };

struct FQAsSubMul {
    size_t output_channels;
    std::vector<int8_t> zp_per_channel;
    std::vector<float> scale_per_channel;
    int8_t zp_min;
    int8_t zp_max;
    float scale_min;
    float scale_max;
};

// Test parameters: <test_case, weight_type, float_element_type, conv_type, target_device>
using NewWeightTablesConvParams = std::tuple<FQAsSubMul, ov::element::Type, ov::element::Type, ConvType, std::string>;

namespace {
template <typename T>
std::vector<T> generateRange(const size_t num_of_elements, const T min_val, const T max_val) {
    std::vector<T> result;
    result.reserve(num_of_elements);

    if (min_val == max_val) {
        // Constant value case
        result.assign(num_of_elements, min_val);
    } else {
        // Distribute values evenly across the range
        for (size_t i = 0; i < num_of_elements; ++i) {
            T value = min_val + static_cast<T>((max_val - min_val) * i / (num_of_elements - 1));
            result.push_back(value);
        }
    }
    return result;
}

FQAsSubMul createTest(const size_t num_channels, const int8_t zp_min, const int8_t zp_max, const float scale_min,
                      const float scale_max) {
    FQAsSubMul result;
    result.output_channels = num_channels;
    result.zp_min = zp_min;
    result.zp_max = zp_max;
    result.scale_min = scale_min;
    result.scale_max = scale_max;
    result.zp_per_channel = generateRange<int8_t>(num_channels, zp_min, zp_max);
    result.scale_per_channel = generateRange<float>(num_channels, scale_min, scale_max);

    return result;
}
}  // namespace

class NewWeightTablesConvTest : public testing::WithParamInterface<NewWeightTablesConvParams>, public VpuOv2LayerTest {
public:
    void SetUp() override {
        const auto& [test_case, weight_type, float_element_type, conv_type, device] = GetParam();
        targetDevice = device;

        size_t inputChannels;
        ov::Shape weightsShape;

        ov::Shape zpShape;
        ov::Shape scaleShape;

        if (conv_type == ConvType::Depthwise) {
            // Depthwise convolution: input channels = output channels = groups
            inputChannels = test_case.output_channels;
            weightsShape = {test_case.output_channels, 1, 1, 1,
                            1};  // [groups, channels_out_per_group, channels_in_per_group, kernel_h, kernel_w]
            zpShape = {test_case.output_channels, 1, 1, 1, 1};
            scaleShape = {test_case.output_channels, 1, 1, 1, 1};
        } else {
            // Regular convolution
            inputChannels = 64;
            weightsShape = {test_case.output_channels, inputChannels, 1, 1};
            zpShape = {test_case.output_channels, 1, 1, 1};
            scaleShape = {test_case.output_channels, 1, 1, 1};
        }

        const size_t totalWeights = weightsShape[0] * weightsShape[1] * weightsShape[2] * weightsShape[3] *
                                    (weightsShape.size() > 4 ? weightsShape[4] : 1);
        // Generates values in range [1, 3] to exercise u2, i4 and i8 weight types with zero-point tables
        std::vector<int8_t> weights = generateRange<int8_t>(totalWeights, /*min_val=*/1, /*max_val=*/3);

        auto i_weights = std::make_shared<ov::op::v0::Constant>(weight_type, weightsShape, weights);
        auto f_weights = std::make_shared<ov::opset6::Convert>(i_weights, float_element_type);

        // Convert integer zero-points to floating-point type, create per-channel zero point tensor and subtract it from
        // weights
        auto i_zp = std::make_shared<ov::op::v0::Constant>(weight_type, zpShape, test_case.zp_per_channel);
        auto f_zp = std::make_shared<ov::opset6::Convert>(i_zp, float_element_type);
        std::shared_ptr<ov::opset6::Subtract> zeroPointsSubtracted =
                std::make_shared<ov::opset6::Subtract>(f_weights, f_zp);

        // Create per-channel scale tensor and multiply zeroPointsSubtracted by it
        auto scale =
                std::make_shared<ov::op::v0::Constant>(float_element_type, scaleShape, test_case.scale_per_channel);
        std::shared_ptr<ov::opset6::Multiply> multiplied =
                std::make_shared<ov::opset6::Multiply>(zeroPointsSubtracted, scale);

        ov::Shape inputShape = {1, inputChannels, 1, 1};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(float_element_type, ov::Shape(inputShape))};

        std::shared_ptr<ov::Node> conv;
        if (conv_type == ConvType::Depthwise) {
            // Depthwise Convolution: [1, C, 1, 1] * [C, 1, 1, 1, 1] = [1, C, 1, 1]
            conv = std::make_shared<ov::op::v1::GroupConvolution>(params[0], multiplied->output(0),
                                                                  ov::Strides{1, 1},         // strides
                                                                  ov::CoordinateDiff{0, 0},  // pads_begin
                                                                  ov::CoordinateDiff{0, 0},  // pads_end
                                                                  ov::Strides{1, 1}          // dilations
            );
        } else {
            // Regular Convolution: [1, 64, 1, 1] * [C, 64, 1, 1] = [1, C, 1, 1]
            conv = std::make_shared<ov::op::v1::Convolution>(params[0], multiplied->output(0),
                                                             ov::Strides{1, 1},         // strides
                                                             ov::CoordinateDiff{0, 0},  // pads_begin
                                                             ov::CoordinateDiff{0, 0},  // pads_end
                                                             ov::Strides{1, 1}          // dilations
            );
        }
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        const std::string model_name =
                conv_type == ConvType::Depthwise ? "NewWeightTables_DWConvolution" : "NewWeightTables_Convolution";
        function = std::make_shared<ov::Model>(results, params, model_name);
    }

    static std::string getTestCaseName(testing::TestParamInfo<NewWeightTablesConvParams> obj) {
        const auto& [test_case, weight_type, precision, conv_type, _] = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << (conv_type == ConvType::Depthwise ? "DWConv" : "Conv") << sep;
        result << "WeightType=" << weight_type.get_type_name() << sep;
        result << "OC=" << test_case.output_channels << sep;
        result << "ZP_" << static_cast<int>(test_case.zp_min);
        if (test_case.zp_min != test_case.zp_max) {
            result << "_to_" << static_cast<int>(test_case.zp_max);
        }
        result << sep;
        result << "precision=" << precision.get_type_name() << sep;
        return result.str();
    }
};

//
// Platform test definition
//

// clang-format off
const auto testCases = std::vector<FQAsSubMul>{
    // 80 channels: ZP in range [1, 3], scale = 1
    createTest(80, 1, 3, 1.0f, 1.0f),
    // 80 channels: ZP in range [-3, 3], scales in range [1, 20]
    createTest(80, -3, 3, 1.0f, 20.0f),
    // 80 channels: ZP = 0, scales in range [1.3, 3.0]
    createTest(80, 0, 0, 1.0f, 1.0f),
    // 80 channels: ZP = 1, scales in range [1.3, 3.0]
    createTest(80, 1, 1, 1.0f, 1.0f),

    // 16 channels: ZP in range [1, 3], scale = 1
    createTest(16, 1, 3, 1.0f, 1.0f),
    // 16 channels: ZP in range [-3, 3], scales in range [1, 20]
    createTest(16, -3, 3, 1.0f, 20.0f),
    // 16 channels: ZP = 0, scales in range [1.3, 3.0]
    createTest(16, 0, 0, 1.0f, 1.0f),
    // 16 channels: ZP = 1, scales in range [1.3, 3.0]
    createTest(16, 1, 1, 1.0f, 1.0f),

    // 20000 channels: ZP in range [1, 5], scale = 1
    createTest(20000, 1, 5, 1.0f, 1.0f),
    // 20000 channels: ZP in range [-5, 5], scale = 1
    createTest(20000, -5, 5, 1.0f, 20.0f),
    // 20000 channels: ZP in range [-5, 5], scales in range [1.3, 3.0]
    createTest(20000, -6, 6, 1.3f, 3.0f),
    // 20000 channels: ZP = 0, scales in range [1.3, 3.0]
    createTest(20000, 0, 0, 1.0f, 1.0f),
    // 20000 channels: ZP = 1, scales in range [1.3, 3.0]
    createTest(20000, 1, 1, 1.0f, 1.0f),
};

const auto u2TestCases = std::vector<FQAsSubMul>{
    // 80 channels: ZP in range [1, 3], scale = 1
    createTest(80, 1, 3, 1.0f, 1.0f),
    // 80 channels: ZP = 0, scales in range [1.3, 3.0]
    createTest(80, 0, 0, 1.0f, 1.0f),
    // 80 channels: ZP = 1, scales in range [1.3, 3.0]
    createTest(80, 1, 1, 1.0f, 1.0f),

    // 16 channels: ZP in range [1, 3], scale = 1
    createTest(16, 1, 3, 1.0f, 1.0f),
    // 16 channels: ZP = 0, scales in range [1.3, 3.0]
    createTest(16, 0, 0, 1.0f, 1.0f),
    // 16 channels: ZP = 1, scales in range [1.3, 3.0]
    createTest(16, 1, 1, 1.0f, 1.0f),

    // 20000 channels: ZP in range [1, 3], scale = 1
    createTest(20000, 1, 3, 1.0f, 1.0f),
    // 20000 channels: ZP = 0, scales in range [1.3, 3.0]
    createTest(20000, 0, 0, 1.0f, 1.0f),
    // 20000 channels: ZP = 1, scales in range [1.3, 3.0]
    createTest(20000, 1, 1, 1.0f, 1.0f),
};

const auto dataPointerTableCases = ::testing::Combine(
        ::testing::ValuesIn(testCases),
        ::testing::Values(ov::element::i8),  // Weight type
        ::testing::Values(ov::element::f16, ov::element::f32), // Computation precision
        ::testing::Values(ConvType::Depthwise),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto zeroPointTableCases = ::testing::Combine(
        ::testing::ValuesIn(testCases),
        ::testing::Values(ov::element::i4, ov::element::i8),  // Weight types
        ::testing::Values(ov::element::f16, ov::element::f32), // Computation precision
        ::testing::Values(ConvType::Regular),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto u2ZeroPointTableCases = ::testing::Combine(
        ::testing::ValuesIn(u2TestCases),
        ::testing::Values(ov::element::u2),  // Weight types
        ::testing::Values(ov::element::f16, ov::element::f32), // Computation precision
        ::testing::Values(ConvType::Regular),
        ::testing::Values(test_utils::TARGET_DEVICE));
// clang-format on

INSTANTIATE_TEST_SUITE_P(zeroPointTable, NewWeightTablesConvTest, zeroPointTableCases,
                         NewWeightTablesConvTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(u2ZeroPointTable, NewWeightTablesConvTest, u2ZeroPointTableCases,
                         NewWeightTablesConvTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(dataPointerTable_Depthwise, NewWeightTablesConvTest, dataPointerTableCases,
                         NewWeightTablesConvTest::getTestCaseName);
