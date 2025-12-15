// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_update.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class ScatterUpdateLayerTestCommon : public ScatterUpdateLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(ScatterUpdateLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterUpdateLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterUpdateLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(ScatterUpdateLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::ScatterUpdateLayerTestCommon;

namespace {
// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
        {{10, 16, 12, 15}, {{{8}, {0, -2}}}}};

std::vector<ov::test::axisUpdateShapeInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisUpdateShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        auto src_shape = input_shape.first;
        auto srcRank = src_shape.size();
        for (auto& item : input_shape.second) {
            auto indices_shape = item.first;
            auto indices_rank = indices_shape.size();
            for (auto& axis : item.second) {
                auto axisP = axis < 0 ? axis + srcRank : axis;
                std::vector<size_t> update_shape;
                for (size_t rs = 0; rs < srcRank; rs++) {
                    if (rs != axisP) {
                        update_shape.push_back(src_shape[rs]);
                    } else {
                        for (size_t ri = 0; ri < indices_rank; ri++) {
                            update_shape.push_back(indices_shape[ri]);
                        }
                    }
                }
                std::vector<ov::Shape> in_shapes{src_shape, update_shape};
                res_vec.push_back(ov::test::axisUpdateShapeInShape{
                        ov::test::static_shapes_to_test_representation(in_shapes), ov::Shape{indices_shape}, axis});
            }
        }
    }
    return res_vec;
}

const std::vector<std::vector<int64_t>> scatterIndices = {{0, 2, 4, 6, 1, 3, 5, 7}};
const auto params = testing::Combine(testing::ValuesIn(combineShapes(axesShapeInShape)),
                                     testing::ValuesIn(scatterIndices), testing::Values(ov::element::f16),
                                     testing::Values(ov::element::i32), testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ScatterUpdate, ScatterUpdateLayerTestCommon, params,
                         ScatterUpdateLayerTestCommon::getTestCaseName);

}  // namespace
