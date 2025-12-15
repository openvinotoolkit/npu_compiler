// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_elements_update.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class ScatterElementsUpdateLayerTestCommon : public ScatterElementsUpdateLayerTest, virtual public VpuOv2LayerTest {};
class ScatterElementsUpdate12LayerTestCommon :
        public ScatterElementsUpdate12LayerTest,
        virtual public VpuOv2LayerTest {};

TEST_P(ScatterElementsUpdateLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterElementsUpdateLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ScatterElementsUpdate12LayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterElementsUpdate12LayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(ScatterElementsUpdateLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(ScatterElementsUpdate12LayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::ScatterElementsUpdate12LayerTestCommon;
using ov::test::ScatterElementsUpdateLayerTestCommon;

namespace {

std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
        {{2, 3, 4}, {{{1, 3, 1}, {1, -1}}}}};

const std::vector<std::vector<size_t>> indicesValue = {{1, 0, 1}};

std::vector<ov::test::axisShapeInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            for (auto& elt : item.second) {
                res_vec.push_back(ov::test::axisShapeInShape{
                        ov::test::static_shapes_to_test_representation({input_shape.first, item.first}), elt});
            }
        }
    }
    return res_vec;
}

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate, ScatterElementsUpdateLayerTestCommon,
                         testing::Combine(testing::ValuesIn(combineShapes(axesShapeInShape)),
                                          testing::ValuesIn(indicesValue), testing::Values(ov::element::f16),
                                          testing::Values(ov::element::i32),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         ScatterElementsUpdateLayerTestCommon::getTestCaseName);

}  // namespace

namespace {  // ScatterElementsUpdate12

std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShapeV12{
        {{2, 3, 4}, {{{1, 7, 1}, {1, -1}}}},
};

const std::vector<std::vector<int64_t>> idxWithNegativeValues = {
        {-1, 0, -1, -1, 2, 1, 1},
};

std::vector<ov::op::v12::ScatterElementsUpdate::Reduction> reductionModes{
        ov::op::v12::ScatterElementsUpdate::Reduction::NONE, ov::op::v12::ScatterElementsUpdate::Reduction::SUM,
        ov::op::v12::ScatterElementsUpdate::Reduction::PROD, ov::op::v12::ScatterElementsUpdate::Reduction::MAX,
        ov::op::v12::ScatterElementsUpdate::Reduction::MIN,  ov::op::v12::ScatterElementsUpdate::Reduction::MEAN,
};

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate12, ScatterElementsUpdate12LayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(combineShapes(axesShapeInShapeV12)),
                                            ::testing::ValuesIn(idxWithNegativeValues),
                                            ::testing::ValuesIn(reductionModes), ::testing::ValuesIn({true, false}),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::element::i32),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ScatterElementsUpdate12LayerTestCommon::getTestCaseName);

}  // namespace
