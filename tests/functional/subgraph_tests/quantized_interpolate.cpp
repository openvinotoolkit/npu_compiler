//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "openvino/op/interpolate.hpp"

using namespace ov::test;
namespace {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//       [input]
//          |
//     (dequantize)
//          |
//        (interp)
//          |
//       [output]
//          |
//      (quantize)
//

using QuantizedSEInterpTestParams = std::tuple<ov::element::Type,                           // inPrc
                                               ov::element::Type,                           // outPrc
                                               std::vector<float>,                          // fqRanges
                                               std::vector<ov::test::InputShape>,           // inputShape
                                               std::vector<float>,                          // scales
                                               ov::op::v4::Interpolate::InterpolateAttrs>;  // interpolate attr
class QuantizedSEInterpSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedSEInterpTestParams> {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-se-ptrs-operations=true";
    }

    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::sin(i);
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void SetUp() override {
        auto [inType, outType, dataFQRanges, inputShape, interpScales, interpolate4Attr] = GetParam();

        init_input_shapes(inputShape);

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {dataFQRanges.at(0)};
        const std::vector<float> dataInHigh = {dataFQRanges.at(1)};
        const std::vector<float> dataOutLow = {dataFQRanges.at(2)};
        const std::vector<float> dataOutHigh = {dataFQRanges.at(3)};
        const auto dataFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, dataInLow,
                                                                dataInHigh, dataOutLow, dataOutHigh);

        auto default_out_shape_node = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 0, 0, 0});
        auto axes_node = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3});
        auto scales_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4}, interpScales);

        auto interp = std::make_shared<ov::op::v4::Interpolate>(dataFq, default_out_shape_node, scales_node, axes_node,
                                                                interpolate4Attr);

        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {255.0f};
        const auto outFq = ov::test::utils::make_fake_quantize(interp, ov::element::f32, dataLevels, {}, outDataLow,
                                                               outDataHigh, outDataLow, outDataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedInterp");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedSEInterpTestParams> obj) {
        auto [ip, op, fqRanges, inputShape, interpScales, interpolate4Attr] = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "FQ=" << vectorToString(fqRanges) << sep;
        result << "InputShape=" << inputShape[0].second[0] << sep;
        result << "InterpScales=" << vectorToString(interpScales) << sep;
        result << "Mode=" << ov::as_string(interpolate4Attr.mode) << sep;
        result << "CoordMode=" << ov::as_string(interpolate4Attr.coordinate_transformation_mode) << sep;
        return result.str();
    }
};

class QuantizedSEInterpSubGraphTest_NPU3720_HW : public QuantizedSEInterpSubGraphTestCommon {};
class QuantizedSEInterpSubGraphTest_NPU3720_SW : public QuantizedSEInterpSubGraphTestCommon {};

TEST_P(QuantizedSEInterpSubGraphTest_NPU3720_HW, HW) {
    abs_threshold = 1.0;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(QuantizedSEInterpSubGraphTest_NPU3720_SW, SW) {
    abs_threshold = 1.0;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

class QuantizedSEInterpSubGraphTest_NPU4000_HW : public QuantizedSEInterpSubGraphTestCommon {};

TEST_P(QuantizedSEInterpSubGraphTest_NPU4000_HW, HW) {
    abs_threshold = 1.0;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

class QuantizedSEInterpSubGraphTest_NPU5010_HW : public QuantizedSEInterpSubGraphTestCommon {};

TEST_P(QuantizedSEInterpSubGraphTest_NPU5010_HW, HW) {
    abs_threshold = 1.0;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

std::vector<std::vector<float>> fqRanges = {{0.0f, 255.0f, 0.0f, 255.0f}};

std::vector<std::vector<ov::Shape>> inputShapes = {{{1, 16, 16, 16}}, {{1, 32, 40, 40}}, {{1, 160, 40, 40}}};

std::vector<std::vector<float>> interpScales = {
        {1.0f, 1.0f, 2.0f, 2.0f}, {1.0f, 1.0f, 3.0f, 3.0f}, {1.0f, 1.0f, 4.0f, 4.0f}};

const std::vector<ov::element::Type> netInPrecisions = {ov::element::f16};

const std::vector<ov::element::Type> netOutputPrecisions = {ov::element::f16};

const std::vector<ov::op::v4::Interpolate::InterpolateAttrs> interpAttrs = {
        ov::op::v4::Interpolate::InterpolateAttrs(ov::op::v4::Interpolate::InterpolateMode::NEAREST,
                                                  ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
                                                  std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
                                                  ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                                  ov::op::v4::Interpolate::NearestMode::FLOOR, false, -0.75),
        ov::op::v4::Interpolate::InterpolateAttrs(ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
                                                  ov::op::v4::Interpolate::ShapeCalcMode::SCALES,
                                                  std::vector<size_t>{0, 0, 0, 0}, std::vector<size_t>{0, 0, 0, 0},
                                                  ov::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                                                  ov::op::v4::Interpolate::NearestMode::FLOOR, false, -0.75)};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(netInPrecisions),
                                           ::testing::ValuesIn(netOutputPrecisions), ::testing::ValuesIn(fqRanges),
                                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                           ::testing::ValuesIn(interpScales), ::testing::ValuesIn(interpAttrs));

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedInterp_HW, QuantizedSEInterpSubGraphTest_NPU3720_HW, basicCases,
                         QuantizedSEInterpSubGraphTest_NPU3720_HW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedInterp_SW, QuantizedSEInterpSubGraphTest_NPU3720_SW, basicCases,
                         QuantizedSEInterpSubGraphTest_NPU3720_SW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedInterp_HW, QuantizedSEInterpSubGraphTest_NPU4000_HW, basicCases,
                         QuantizedSEInterpSubGraphTest_NPU4000_HW::getTestCaseName);

// TODO: E156484 - investigate accuracy issue
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_QuantizedInterp_HW, QuantizedSEInterpSubGraphTest_NPU5010_HW, basicCases,
                         QuantizedSEInterpSubGraphTest_NPU5010_HW::getTestCaseName);

// TODO: E156484 - investigate accuracy issue

}  // namespace
