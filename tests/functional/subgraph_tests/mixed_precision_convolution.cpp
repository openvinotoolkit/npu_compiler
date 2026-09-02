//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mixed_precision_convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/core/type/float4_e2m1.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/multiply.hpp"

#include <algorithm>
#include <random>

namespace ov {
namespace test {

class MixedPrecisionConvSubGraphTestCommon : public MixedPrecisionConvSubGraphTest {};

TEST_P(MixedPrecisionConvSubGraphTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MixedPrecisionConvSubGraphTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

using MixedPrecisionConvSubGraphTestNF4 = MixedPrecisionConvSubGraphTestCommon;
TEST_P(MixedPrecisionConvSubGraphTestNF4, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

// Test for Convolution with FP8 fake-quantized activations and FP4 dequantized weights
class MixedPrecisionConvSubGraphTestFP8ActFP4Weights : public MixedPrecisionConvSubGraphTest {
public:
    void SetUp() override {
        abs_threshold = 1.0f;

        mixedPrecisionConvSpecificParams mixedPrecisionConvParams;
        ov::element::Type modelType;
        std::vector<size_t> inputShape;
        std::tie(mixedPrecisionConvParams, modelType, inputShape, std::ignore) = this->GetParam();

        ov::op::PadType padType = ov::op::PadType::AUTO;
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, std::ignore, std::ignore, std::ignore) =
                mixedPrecisionConvParams;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        // Activation: input fake-quantized to f8e4m3 storage via FakeConvert.
        // FakeConvert(x, scale) = round_fp8(x / scale) * scale; scale=2.0 exercises
        // the scale path and doubles the representable FP8 range (max: 2 * 448 = 896).
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front())};
        const auto actScale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1}, {2.0f});
        const auto actFakeConvert =
                std::make_shared<ov::op::v13::FakeConvert>(params[0], actScale, ov::element::f8e4m3);

        // Weights: f4e2m1 constants dequantized to f16 via Convert + per-output-channel Multiply.
        const std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1], kernel[0], kernel[1]};
        static const std::vector<float> fp4Vals = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f, 6.0f,
                                                   -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        std::mt19937 fp4Engine(0);
        std::uniform_int_distribution<> fp4IdxDist(0, static_cast<int>(fp4Vals.size()) - 1);
        std::vector<ov::float4_e2m1> fp4Data(ov::shape_size(weightsShapes));
        std::generate(fp4Data.begin(), fp4Data.end(), [&]() {
            return ov::float4_e2m1(fp4Vals[fp4IdxDist(fp4Engine)]);
        });
        const auto weightsConst = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, weightsShapes, fp4Data);
        const auto weightsConvert = std::make_shared<ov::op::v0::Convert>(weightsConst, ov::element::f16);
        // Per-output-channel scale: 1/6.0 normalises FP4 max value (6.0) to 1.0.
        const std::vector<double> scaleData(convOutChannels, 1.0 / 6.0);
        const auto scaleConst = std::make_shared<ov::op::v0::Constant>(ov::element::f16,
                                                                       ov::Shape{convOutChannels, 1, 1, 1}, scaleData);
        const auto weightsMultiply = std::make_shared<ov::op::v1::Multiply>(weightsConvert, scaleConst);

        auto conv = std::make_shared<ov::op::v1::Convolution>(actFakeConvert, weightsMultiply, stride, padBegin, padEnd,
                                                              dilation, padType);
        function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(conv)}, params,
                                               "MixedPrecisionConvFP8ActFP4Weights");
    }
};

TEST_P(MixedPrecisionConvSubGraphTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(MixedPrecisionConvSubGraphTestNF4, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
using namespace ov::test::utils;

namespace {

const auto conv2DParamsI8 =
        ::testing::Combine(::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padEnds
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // dilations
                           ::testing::Values(16),                                    // numOutChannels
                           ::testing::Values(LowFpType::Undefined),                  // lowFpType
                           ::testing::Values(255),                                   // quantLevels
                           ::testing::Values(QuantizationGranularity::Pertensor)     // quantGranularity
        );

const auto conv2DParamsI4 =
        ::testing::Combine(::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padEnds
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // dilations
                           ::testing::Values(16),                                    // numOutChannels
                           ::testing::Values(LowFpType::Undefined),                  // lowFpType
                           ::testing::Values(16),                                    // quantLevels
                           ::testing::Values(QuantizationGranularity::Pertensor)     // quantGranularity
        );

const auto conv2DParamsNF4 =
        ::testing::Combine(::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padEnds
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // dilations
                           ::testing::Values(16),                                    // numOutChannels
                           ::testing::Values(LowFpType::NF4),                        // lowFpType
                           ::testing::Values(0),                                     // quantLevels
                           ::testing::Values(QuantizationGranularity::Pertensor)     // quantGranularity
        );

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I8, MixedPrecisionConvSubGraphTestCommon,
                         ::testing::Combine(conv2DParamsI8,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I4, MixedPrecisionConvSubGraphTestCommon,
                         ::testing::Combine(conv2DParamsI4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTestCommon::getTestCaseName);

// nf4 test cases (NPU4000+)
INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_NF4, MixedPrecisionConvSubGraphTestNF4,
                         ::testing::Combine(conv2DParamsNF4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTestNF4::getTestCaseName);

}  // namespace
