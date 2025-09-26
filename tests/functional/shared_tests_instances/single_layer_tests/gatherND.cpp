// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/node_builders/gather_nd.hpp"
#include "openvino/op/gather_nd.hpp"
#include "single_op_tests/gather_nd.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

// [E#177479] GatherND operator is limited to positive indices as it does not support negative indices at this time
class GatherNDLayerTestCommon : public GatherND8LayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        ov::Shape indicesShape;
        std::vector<InputShape> shapes;
        ov::element::Type modelType, indicesType;
        int batch;
        std::tie(shapes, indicesShape, batch, modelType, indicesType, targetDevice) = this->GetParam();
        init_input_shapes(shapes);

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front());

        auto batchDim = targetStaticShapes[0][0][0];
        ov::test::utils::InputGenerateData inData;

        inData.start_from = 0;
        inData.range = batchDim - 1;

        auto indicesNodeTensor = ov::test::utils::create_and_fill_tensor(indicesType, indicesShape, inData);
        auto indicesNode = std::make_shared<ov::op::v0::Constant>(indicesNodeTensor);

        auto gatherND = std::make_shared<ov::op::v8::GatherND>(param, indicesNode, batch);
        gatherND->set_friendly_name("GatherND");

        auto result = std::make_shared<ov::op::v0::Result>(gatherND);

        function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "gatherND");
    }
};

TEST_P(GatherNDLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherNDLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> dPrecisions = {
        ov::element::i32,
};

const std::vector<ov::element::Type> iPrecisions = {
        ov::element::i32,
};

std::vector<std::vector<ov::Shape>> iShapeSubset1 = {{{2, 2}}, {{2, 3, 4}}};
const auto gatherNDArgsSubset1 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShapeSubset1)),  // Data shape
                         testing::ValuesIn(std::vector<ov::Shape>({{2, 1}, {2, 1, 1}})),          // Indices shape
                         testing::ValuesIn(std::vector<int>({0, 1})),                             // Batch dims
                         testing::ValuesIn(dPrecisions),                                          // Model type
                         testing::ValuesIn(iPrecisions),                                          // Indices type
                         testing::Values(test_utils::TARGET_DEVICE));                             // Device name

std::vector<std::vector<ov::Shape>> iShapeSubsetPrecommit = {{{5, 7, 3}}};
const auto gatherNDArgsSubsetPrecommit =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShapeSubsetPrecommit)),
                         testing::ValuesIn(std::vector<ov::Shape>({{5, 1}})), testing::ValuesIn(std::vector<int>({1})),
                         testing::Values(ov::element::i32), testing::Values(ov::element::i32),
                         testing::Values(test_utils::TARGET_DEVICE));

std::vector<std::vector<ov::Shape>> iShapeSubsetTiling = {{{1, 16, 32, 56, 16}}};
const auto gatherNDArgsSubsetTiling =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShapeSubsetTiling)),
                         testing::ValuesIn(std::vector<ov::Shape>({{1, 16, 14580, 2}})),
                         testing::ValuesIn(std::vector<int>({2})), testing::Values(ov::element::i32),
                         testing::Values(ov::element::i32), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_GatherND, GatherNDLayerTestCommon, gatherNDArgsSubset1,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherND, GatherNDLayerTestCommon, gatherNDArgsSubsetPrecommit,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_GatherND, GatherNDLayerTestCommon, gatherNDArgsSubsetTiling,
                         GatherND8LayerTest::getTestCaseName);

}  // namespace
