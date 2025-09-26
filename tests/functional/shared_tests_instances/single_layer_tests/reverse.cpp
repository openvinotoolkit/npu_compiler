//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reverse.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/reverse.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {
class ReverseLayerTestCommon : public ReverseLayerTest, public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<size_t> input_shape;
        std::vector<int> axes;
        std::string mode;
        ov::element::Type model_type;

        std::tie(input_shape, axes, mode, model_type, std::ignore) = GetParam();

        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({input_shape}));
        auto param = std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(input_shape));

        std::shared_ptr<ov::op::v0::Constant> axes_constant;
        if (mode == "index") {
            axes_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{axes.size()}, axes);
        } else {
            std::vector<bool> axes_mask(input_shape.size(), false);
            for (auto axis : axes) {
                axes_mask[axis] = true;
            }
            axes_constant = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{axes_mask.size()},
                                                                   axes_mask);
        }

        auto rev = std::make_shared<ov::op::v1::Reverse>(param, axes_constant, mode);
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(rev->outputs(), ov::ParameterVector{param}, "reverse");
    }

    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(ReverseLayerTestCommon, NPU3720_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReverseLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}

TEST_P(ReverseLayerTestCommon, NPU4000_HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> netPrecisions = {ov::element::f16};
const std::vector<std::string> modes = {"index"};

const std::vector<std::vector<size_t>> inputShapes1D = {{7}, {12}};
const std::vector<std::vector<int>> indices1D = {{0}};

const std::vector<std::vector<size_t>> inputShapes2D = {{3, 5}, {5, 2}, {6, 6}};
const std::vector<std::vector<int>> indices2D = {{0}, {1}, {0, 1}};

const std::vector<std::vector<size_t>> inputShapes3D = {{1, 2, 3}, {3, 5, 9}, {6, 4, 5}, {7, 5, 3}};
const std::vector<std::vector<int>> indices3D = {{0}, {1, 2}, {0, 1, 2}};

const std::vector<std::vector<size_t>> inputShapes4D = {{1, 1, 1, 2}, {2, 1, 2, 1}, {3, 2, 1, 1}, {5, 5, 5, 5}};
const std::vector<std::vector<int>> indices4D = {{1}, {0, 1}, {0, 1, 3}, {0, 1, 2, 3}};

const std::vector<std::vector<size_t>> inputShapesMC = {
        {8, 64, 64, 2}, {3, 8, 1, 8}, {6, 1, 8, 8}, {1, 8, 8, 8}, {1, 1, 16, 16}};
const std::vector<std::vector<int>> indicesMC = {{1, 2, 3}, {1, 3}, {0, 2}};

const auto params1D =
        testing::Combine(testing::ValuesIn(inputShapes1D), testing::ValuesIn(indices1D), testing::ValuesIn(modes),
                         testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto params2D =
        testing::Combine(testing::ValuesIn(inputShapes2D), testing::ValuesIn(indices2D), testing::ValuesIn(modes),
                         testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto params3D =
        testing::Combine(testing::ValuesIn(inputShapes3D), testing::ValuesIn(indices3D), testing::ValuesIn(modes),
                         testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto params4D =
        testing::Combine(testing::ValuesIn(inputShapes4D), testing::ValuesIn(indices4D), testing::ValuesIn(modes),
                         testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsMC =
        testing::Combine(testing::ValuesIn(inputShapesMC), testing::ValuesIn(indicesMC), testing::ValuesIn(modes),
                         testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const std::vector<std::vector<size_t>> inputShapesPrecommit1D = {{1}, {2}};
const std::vector<std::vector<int>> indicesPrecommit1D = {{0}};

const std::vector<std::vector<size_t>> inputShapesPrecommit2D = {{2, 4}, {4, 2}};
const std::vector<std::vector<int>> indicesPrecommit2D = {{1}, {0, 1}};

const std::vector<std::vector<size_t>> inputShapesPrecommit3D = {{1, 1, 1}, {1, 1, 2}};
const std::vector<std::vector<int>> indicesPrecommit3D = {{1}, {0, 2}};

const std::vector<std::vector<size_t>> inputShapesPrecommit4D = {{2, 1, 3, 2}, {2, 2, 2, 3}};
const std::vector<std::vector<int>> indicesPrecommit4D = {{2}, {0, 2, 3}, {0, 1, 2, 3}};

const auto paramsPrecommit1D = testing::Combine(
        testing::ValuesIn(inputShapesPrecommit1D), testing::ValuesIn(indicesPrecommit1D), testing::ValuesIn(modes),
        testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsPrecommit2D = testing::Combine(
        testing::ValuesIn(inputShapesPrecommit2D), testing::ValuesIn(indicesPrecommit2D), testing::ValuesIn(modes),
        testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsPrecommit3D = testing::Combine(
        testing::ValuesIn(inputShapesPrecommit3D), testing::ValuesIn(indicesPrecommit3D), testing::ValuesIn(modes),
        testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

const auto paramsPrecommit4D = testing::Combine(
        testing::ValuesIn(inputShapesPrecommit4D), testing::ValuesIn(indicesPrecommit4D), testing::ValuesIn(modes),
        testing::ValuesIn(netPrecisions), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_1D, ReverseLayerTestCommon, params1D, ReverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Reverse_2D, ReverseLayerTestCommon, params2D, ReverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Reverse_3D, ReverseLayerTestCommon, params3D, ReverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Reverse_4D, ReverseLayerTestCommon, params4D, ReverseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reverse_MC, ReverseLayerTestCommon, paramsMC, ReverseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Reverse_1D, ReverseLayerTestCommon, paramsPrecommit1D,
                         ReverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Reverse_2D, ReverseLayerTestCommon, paramsPrecommit2D,
                         ReverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Reverse_3D, ReverseLayerTestCommon, paramsPrecommit3D,
                         ReverseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Reverse_4D, ReverseLayerTestCommon, paramsPrecommit4D,
                         ReverseLayerTest::getTestCaseName);

}  // namespace
