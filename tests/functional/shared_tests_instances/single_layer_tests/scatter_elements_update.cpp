//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_elements_update.hpp"
#include <random>
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ScatterElementsUpdateLayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ScatterElementsUpdate12LayerTest);

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

TEST_P(ScatterElementsUpdateLayerTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_P(ScatterElementsUpdate12LayerTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
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
        // Cover an axis-1 update and an innermost-axis update represented by
        // -1, which is normalized by the operation before kernel dispatch.
        {{2, 3, 4},
         {
                 {{1, 7, 1}, {1}},
                 {{1, 1, 7}, {-1}},
         }},
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

// Generates random indices from the first shape configuration
// Since values are related to input shape, this function only works with single case scenario
std::vector<int64_t> generateConstIndicesForLargeShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes_map) {
    const auto& [input_shape, indices_map] = *input_shapes_map.begin();
    const auto& [indices_shape, axis_list] = *indices_map.begin();
    int64_t index_limit = input_shape[axis_list[0]];
    std::vector<int64_t> indices_value;
    auto indices_element_count = ov::shape_size(indices_shape);
    indices_value.reserve(indices_element_count);
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int64_t> dis(-index_limit, index_limit - 1);
    for (size_t i = 0; i < indices_element_count; ++i) {
        indices_value.push_back(dis(gen));
    }
    return indices_value;
}

std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> inputShapesForLargeShapes{
        {{1024, 2048}, {{{32, 2048}, {0}}}},
};

std::vector<ov::op::v12::ScatterElementsUpdate::Reduction> reductionModesForLargeShapes{
        ov::op::v12::ScatterElementsUpdate::Reduction::SUM,
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ScatterElementsUpdate12ForLargeShapes, ScatterElementsUpdate12LayerTestCommon,
        ::testing::Combine(::testing::ValuesIn(combineShapes(inputShapesForLargeShapes)),
                           ::testing::Values(generateConstIndicesForLargeShapes(inputShapesForLargeShapes)),
                           ::testing::ValuesIn(reductionModesForLargeShapes), ::testing::ValuesIn({true}),
                           ::testing::Values(ov::element::f32), ::testing::Values(ov::element::i32),
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        ScatterElementsUpdate12LayerTestCommon::getTestCaseName);

}  // namespace

namespace {

std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> innermostFastPathShape{
        {{2, 64}, {{{2, 16}, {1}}}},
};

// Dense innermost-axis, reduction=NONE cases exercise direct stores in the
// fp16 fast path, including valid negative indices and both init-value modes.
const std::vector<std::vector<int64_t>> innermostFastPathIndices{
        {3, -1, -64, 5, 10, 15, 20, 7,  8, 12, 22, 30, 40, 50, 60, 2,
         1, 2,  4,   8, 16, 32, -1, -2, 9, 25, 5,  7,  11, 13, 17, 19},
};

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate12_InnermostFp16FastPath, ScatterElementsUpdate12LayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(combineShapes(innermostFastPathShape)),
                                            ::testing::ValuesIn(innermostFastPathIndices),
                                            ::testing::Values(ov::op::v12::ScatterElementsUpdate::Reduction::NONE),
                                            ::testing::ValuesIn({true, false}), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ScatterElementsUpdate12LayerTestCommon::getTestCaseName);

std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> innermostCollapseShape{
        {{2, 3, 8, 1}, {{{2, 1, 8, 1}, {1}}}},
};

// The singleton innermost dimension makes the scatter axis eligible for the
// kernel's axis-1-to-innermost collapse path; negatives and both init modes
// keep normalization and initialization behavior covered.
const std::vector<std::vector<int64_t>> innermostCollapseIndices{
        {0, 1, 2, 0, -1, -2, -3, 1, 2, 0, 1, -1, -3, 2, 0, -2},
};

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate12_InnermostFp16CollapsePath,
                         ScatterElementsUpdate12LayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(combineShapes(innermostCollapseShape)),
                                            ::testing::ValuesIn(innermostCollapseIndices),
                                            ::testing::Values(ov::op::v12::ScatterElementsUpdate::Reduction::NONE),
                                            ::testing::ValuesIn({true, false}), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ScatterElementsUpdate12LayerTestCommon::getTestCaseName);

// Innermost-axis update with a smaller non-axis outer dim that wraps into a
// further outer dim (data [3,5,4], updates/indices [3,2,4], axis=2): the
// dimension right before the innermost axis is legally smaller in updates
// than in data (2 < 5, per spec updates dims may be <= data dims outside
// axis) while the outermost dim (3) is > 1, so the fast path's flat
// pOut += wDim row walk would skip/misalign output addresses if the
// updateShape/outputShape equality guard were missing.
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> innermostWrappingOuterDimShape{
        {{3, 5, 4}, {{{3, 2, 4}, {2}}}},
};

const std::vector<std::vector<int64_t>> innermostWrappingOuterDimIndices{
        {0, -1, 2, -4, 1, 3, -2, -3, 3, 0, -1, 2, -3, -4, 1, 2, 2, 1, -2, 0, -1, 3, -4, 0},
};

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate12_InnermostFp16WrappingOuterDim,
                         ScatterElementsUpdate12LayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(combineShapes(innermostWrappingOuterDimShape)),
                                            ::testing::ValuesIn(innermostWrappingOuterDimIndices),
                                            ::testing::Values(ov::op::v12::ScatterElementsUpdate::Reduction::NONE),
                                            ::testing::ValuesIn({true, false}), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::i32),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ScatterElementsUpdate12LayerTestCommon::getTestCaseName);

}  // namespace
