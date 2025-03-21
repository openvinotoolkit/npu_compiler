// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/utils.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "intel_npu/npu_private_properties.hpp"

using namespace ov::test;
namespace {

// Test pattern
//
//       [input] [const]
//          |
//        (Conv)
//          |
//      (Interpolate)
//          |
//       [Transpose CHW-HWC]
//          |
//      (output)
//
// Scope: Test interpolate Sw kernel in Channel minor.

struct ConvInterpolateTransposeTestParams {
    size_t channelSize;
};

class ConvInterpolateTransposeTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvInterpolateTransposeTestParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "map-interpolate-on-dpu=false";
    }
    void SetUp() override {
        const auto testParams = GetParam();
        size_t channelSize = testParams.channelSize;
        const ov::Shape inputShape = {1, channelSize, 14, 14};
        const ov::Shape outputShape = {1, channelSize, 33, 33};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front());

        const auto weightsShape = ov::Shape{channelSize, channelSize, 1, 1};
        size_t convConstSize = channelSize * channelSize * 1 * 1;
        std::vector<float> convConstData(convConstSize, 0.0f);
        for (std::size_t i = 0; i < channelSize; i++) {
            convConstData.at(i * channelSize + i) = 1.0f;
        }
        const auto weights = ov::op::v0::Constant::create(ov::element::f32, weightsShape, convConstData.data());
        const auto weightsFP16 = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f16);

        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                param, weightsFP16->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        auto interpolateConst0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, outputShape.data());
        auto interpolateConst1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4},
                                                              {1.000000e+00, 1.000000e+00, 2.3571428571, 2.3571428571});

        auto interpolateAttr = ov::op::v4::Interpolate::InterpolateAttrs(
                ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX, ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
                std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
                ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR, false, -0.75);

        auto interp = std::make_shared<ov::op::v4::Interpolate>(conv2d_node, interpolateConst0, interpolateConst1,
                                                                interpolateAttr);

        auto transposeConst0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});

        auto transpose = std::make_shared<ov::op::v1::Transpose>(interp, transposeConst0);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose->output(0))};
        ov::ParameterVector params{param};
        function = std::make_shared<ov::Model>(results, params, "ConvInterpolateTranspose");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvInterpolateTransposeTestParams> obj) {
        size_t channelSize = obj.param.channelSize;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "channelSize={" << channelSize << "}";
        return result.str();
    }
};

TEST_P(ConvInterpolateTransposeTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
TEST_P(ConvInterpolateTransposeTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
const std::vector<ConvInterpolateTransposeTestParams> cSzParams = {{1},  {3},  {4},  {7},  {8}, {9},
                                                                   {16}, {21}, {32}, {33}, {64}};

INSTANTIATE_TEST_SUITE_P(smoke_convInterpolateTransposeTest, ConvInterpolateTransposeTestCommon,
                         ::testing::ValuesIn(cSzParams), ConvInterpolateTransposeTestCommon::getTestCaseName);

}  // namespace
