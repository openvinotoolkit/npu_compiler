//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grid_sample.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GridSampleLayerTestCommon : public GridSampleLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        auto itTargetShape = targetInputStaticShapes.begin();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            tensor = create_and_fill_tensor(funcInput.get_element_type(), *itTargetShape, 0, 0.5);

            VpuOv2LayerTest::inputs.insert({funcInput.get_node_shared_ptr(), tensor});
            itTargetShape++;
        }
    }
    void SetUp() override {
        ov::Shape dataShape, gridShape;
        bool alignCorners;
        ov::op::v9::GridSample::InterpolationMode mode;
        ov::op::v9::GridSample::PaddingMode paddingMode;
        ov::element::Type modelType, gridType;

        std::tie(dataShape, gridShape, alignCorners, mode, paddingMode, modelType, gridType, std::ignore) =
                this->GetParam();

        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({dataShape, gridShape}));

        auto data = std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes[0]);
        // C#133057
        // `grid` element type should not be hardcoded to `f32`
        auto grid = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, VpuOv2LayerTest::inputDynamicShapes[1]);
        auto gridSample = std::make_shared<ov::op::v9::GridSample>(
                data, grid, ov::op::v9::GridSample::Attributes(alignCorners, mode, paddingMode));

        VpuOv2LayerTest::function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(gridSample),
                                                                ov::ParameterVector{data, grid}, "GridSample");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(GridSampleLayerTestCommon, NPU3720_HW) {
    VpuOv2LayerTest::abs_threshold = 0.8;
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(GridSampleLayerTestCommon, NPU4000_HW) {
    VpuOv2LayerTest::abs_threshold = 0.8;
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

using GridSampleOp = ov::op::v9::GridSample;

namespace {

const std::vector<ov::Shape> dataShapes = {{2, 2, 3, 4}};

const std::vector<ov::Shape> gridShapes = {{2, 2, 3, 2}};

const std::vector<ov::Shape> dataShapesTiling = {{1, 2, 800, 800}};

const std::vector<ov::Shape> gridShapesTiling = {{1, 2, 2, 2}};

const std::vector<bool> alignCorners = {true, false};

const std::vector<GridSampleOp::InterpolationMode> modes = {
        GridSampleOp::InterpolationMode::BILINEAR,
        GridSampleOp::InterpolationMode::NEAREST,
        GridSampleOp::InterpolationMode::BICUBIC,
};

const std::vector<GridSampleOp::PaddingMode> paddingModes = {
        GridSampleOp::PaddingMode::ZEROS, GridSampleOp::PaddingMode::BORDER, GridSampleOp::PaddingMode::REFLECTION};

const std::vector<ov::element::Type> dataTypes = {
        ov::element::f16,
};

const std::vector<ov::element::Type> gridTypes = {
        ov::element::f16,
};

const auto params = testing::Combine(::testing::ValuesIn(dataShapes), ::testing::ValuesIn(gridShapes),
                                     ::testing::ValuesIn(alignCorners), ::testing::ValuesIn(modes),
                                     ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataTypes),
                                     ::testing::ValuesIn(gridTypes), ::testing::Values(test_utils::TARGET_DEVICE));

const auto paramsTiling = testing::Combine(
        ::testing::ValuesIn(dataShapesTiling), ::testing::ValuesIn(gridShapesTiling), ::testing::ValuesIn(alignCorners),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(paddingModes), ::testing::ValuesIn(dataTypes),
        ::testing::ValuesIn(gridTypes), ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample, GridSampleLayerTestCommon, params,
                         GridSampleLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GridSample_Tiling, GridSampleLayerTestCommon, paramsTiling,
                         GridSampleLayerTest::getTestCaseName);

}  // namespace
