// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/psroi_pooling.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/psroi_pooling.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class PSROIPoolingLayerTestCommon : public PSROIPoolingLayerTest, public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<size_t> inputShapes, coordsShape;
        size_t outputDim, groupSize, spatialBinsX, spatialBinsY;
        float spatialScale;
        std::string mode;
        ov::element::Type modelType;
        std::tie(inputShapes, coordsShape, outputDim, groupSize, spatialScale, spatialBinsX, spatialBinsY, mode,
                 modelType, std::ignore) = this->GetParam();

        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShapes, coordsShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes[0]),
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes[1])};
        auto psroiPooling = std::make_shared<ov::op::v0::PSROIPooling>(params[0], params[1], outputDim, groupSize,
                                                                       spatialScale, spatialBinsX, spatialBinsY, mode);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(psroiPooling->outputs(), params, "psroiPooling");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(PSROIPoolingLayerTestCommon, NPU3720_HW) {
    VpuOv2LayerTest::abs_threshold = 0.016f;
    VpuOv2LayerTest::setSkipCompilationCallback([this](std::stringstream& skip) {
        std::string psROIPoolingMode = std::get<7>(GetParam());
        if (psROIPoolingMode == "bilinear") {
            skip << "BILINEAR mode is unsupported for now";
        }
    });
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(PSROIPoolingLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::abs_threshold = 0.016f;
    VpuOv2LayerTest::setSkipCompilationCallback([this](std::stringstream& skip) {
        std::string psROIPoolingMode = std::get<7>(GetParam());
        if (psROIPoolingMode == "bilinear") {
            skip << "BILINEAR mode is unsupported for now";
        }
    });
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
TEST_P(PSROIPoolingLayerTestCommon, NPU5010_SW) {
    VpuOv2LayerTest::abs_threshold = 0.016f;
    VpuOv2LayerTest::setSkipCompilationCallback([this](std::stringstream& skip) {
        std::string psROIPoolingMode = std::get<7>(GetParam());
        if (psROIPoolingMode == "bilinear") {
            skip << "BILINEAR mode is unsupported for now";
        }
    });
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

const std::vector<ov::element::Type> modelTypes = {ov::element::f32, ov::element::f16};

const std::vector<std::vector<size_t>> inputShapeVector0 = {
        {2, 200, 20, 20}, {2, 200, 20, 16}, {2, 200, 16, 20}, {3, 200, 16, 16}};
const std::vector<std::vector<size_t>> inputShapeVector1 = {{1, 49 * 1, 14, 14}};
const std::vector<std::vector<size_t>> inputShapeVector2 = {{1, 3240, 38, 38}};

const std::vector<std::vector<size_t>> coordShapesVector0 = {{1, 5}};
const std::vector<std::vector<size_t>> coordShapesVector1 = {{300, 5}};
const std::vector<std::vector<size_t>> coordShapesVector2 = {{100, 5}};

const auto paramsAvg0 = testing::Combine(::testing::ValuesIn(inputShapeVector0),   // input
                                         ::testing::ValuesIn(coordShapesVector0),  // coord
                                         ::testing::Values(50),                    // outputDim
                                         ::testing::Values(2),                     // groupSize
                                         ::testing::Values(1.0f),                  // spatialScale
                                         ::testing::Values(1),                     // spatialBinX
                                         ::testing::Values(1),                     // spatialBinY
                                         ::testing::Values("average"),             // mode
                                         ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramsAvg1 = testing::Combine(::testing::ValuesIn(inputShapeVector1),   // input
                                         ::testing::ValuesIn(coordShapesVector1),  // coord
                                         ::testing::Values(1),                     // outputDim
                                         ::testing::Values(7),                     // groupSize
                                         ::testing::Values(0.0625f),               // spatialScale
                                         ::testing::Values(1),                     // spatialBinX
                                         ::testing::Values(1),                     // spatialBinY
                                         ::testing::Values("average"),             // mode
                                         ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramsBilinear =
        testing::Combine(::testing::ValuesIn(inputShapeVector2),   // input
                         ::testing::ValuesIn(coordShapesVector2),  // coord
                         ::testing::Values(360),                   // outputDim
                         ::testing::Values(6),                     // groupSize
                         ::testing::Values(1.0f),                  // spatialScale
                         ::testing::Values(3),                     // spatialBinX
                         ::testing::Values(3),                     // spatialBinY
                         ::testing::Values("bilinear"),            // mode
                         ::testing::ValuesIn(modelTypes), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest0, PSROIPoolingLayerTestCommon, paramsAvg0,
                         PSROIPoolingLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingAverageLayoutTest1, PSROIPoolingLayerTestCommon, paramsAvg1,
                         PSROIPoolingLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_PSROIPoolingBiliniarLayoutTest0, PSROIPoolingLayerTestCommon, paramsBilinear,
                         PSROIPoolingLayerTestCommon::getTestCaseName);
