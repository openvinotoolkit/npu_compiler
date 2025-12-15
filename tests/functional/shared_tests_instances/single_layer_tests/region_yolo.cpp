// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/region_yolo.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "openvino/op/region_yolo.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class RegionYoloLayerTestCommon : public RegionYoloLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<size_t> inputShape;
        ov::element::Type modelType;
        size_t classes, coords, numRegions;
        bool doSoftmax;
        std::vector<int64_t> mask;
        int startAxis, endAxis;
        std::tie(inputShape, classes, coords, numRegions, doSoftmax, mask, startAxis, endAxis, modelType, std::ignore) =
                this->GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front());
        auto regionYolo = std::make_shared<ov::op::v0::RegionYolo>(param, coords, classes, numRegions, doSoftmax, mask,
                                                                   startAxis, endAxis);
        VpuOv2LayerTest::function =
                std::make_shared<ov::Model>(regionYolo->outputs(), ov::ParameterVector{param}, "RegionYolo");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};

TEST_P(RegionYoloLayerTestCommon, NPU3720_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(RegionYoloLayerTestCommon, NPU4000_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}
TEST_P(RegionYoloLayerTestCommon, NPU5010_SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<std::vector<size_t>> inputShapes = {{{1, 125, 13, 13}}};

const std::vector<std::vector<size_t>> inputShapesPrecommit = {{{1, 27, 26, 26}}};

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const auto regionYoloParams =
        ::testing::Combine(testing::ValuesIn(inputShapes),
                           testing::Values(20),                                     // classes
                           testing::Values(4),                                      // coords
                           testing::Values(5),                                      // numRegions
                           testing::Values(false, true),                            // doSoftmax
                           testing::Values(std::vector<int64_t>({0, 1, 2, 3, 4})),  // mask
                           testing::Values(1),                                      // startAxis
                           testing::Values(3),                                      // endAxis
                           testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

const auto regionYoloPrecommitParams =
        ::testing::Combine(testing::ValuesIn(inputShapesPrecommit),
                           testing::Values(4),                                // classes
                           testing::Values(4),                                // coords
                           testing::Values(9),                                // numRegions
                           testing::Values(false),                            // doSoftmax
                           testing::Values(std::vector<int64_t>({0, 1, 2})),  // mask
                           testing::Values(1),                                // startAxis
                           testing::Values(3),                                // endAxis
                           testing::ValuesIn(modelTypes), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_RegionYolo, RegionYoloLayerTestCommon, regionYoloParams,
                         RegionYoloLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_RegionYolo, RegionYoloLayerTestCommon, regionYoloPrecommitParams,
                         RegionYoloLayerTestCommon::getTestCaseName);

}  // namespace
