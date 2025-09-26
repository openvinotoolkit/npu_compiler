// Copyright (C) 2021-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_ND_update.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class ScatterNDUpdateLayerTestCommon : public ScatterNDUpdateLayerTest, virtual public VpuOv2LayerTest {};
class ScatterNDUpdateLayerTestMTLHW : public ScatterNDUpdateLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(ScatterNDUpdateLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

// Special class for tests that work only on MTL and HW pipe
TEST_P(ScatterNDUpdateLayerTestMTLHW, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterNDUpdateLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using ov::test::ScatterNDUpdateLayerTestCommon;
using ov::test::ScatterNDUpdateLayerTestMTLHW;

namespace {

// map<inputShape vector<pair<indicesShape, indicesValue>>>
// updateShape is gotten from inputShape and indicesShape
using InputMap = std::map<std::vector<size_t>, std::vector<std::pair<std::vector<size_t>, std::vector<int>>>>;

InputMap sliceSelectInShape{
        {{1}, {{{1, 1}, {0}}}},
        {{8}, {{{4, 1}, {4, 3, 1, 7}}}},
        {{1, 32, 1},
         {{{1, 3, 1, 3}, {0, 10, 0, 0, 11, 0, 0, 12, 0}},
          {{1, 3, 1, 3}, {0, 0, 0, 0, 1, 0, 0, 2, 0}},
          {{1, 3, 1, 3}, {0, 29, 0, 0, 30, 0, 0, 31, 0}}}},
        {{4, 4, 4}, {{{2, 1}, {0, 2}}, {{2, 1}, {1, 2}}, {{2, 2, 2}, {0, 0, 2, 2, 1, 1, 3, 3}}}},
        {{3, 3, 3},
         {{{2, 1}, {0, 2}},
          {{2, 2, 3}, {0, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 2}},
          {{2, 2}, {0, 0, 2, 2}},
          {{2, 3}, {0, 0, 0, 2, 2, 2}}}},
        {{4, 5, 6}, {{{2, 2, 2, 3}, {1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2}}}},
        {{1, 1, 4, 4},
         {{{1, 1, 2, 2, 4}, {0, 0, 1, 1, 0, 0, 1, 3, 0, 0, 3, 1, 0, 0, 3, 3}},
          {{1, 1, 2, 2, 4}, {0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 2, 1, 0, 0, 2, 3}},
          {{1, 1, 2, 2, 4}, {0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 2}}}}};

InputMap precommit_sliceSelectInShape{
        // {{2, 3}, {{{1, 2}, {1, 3}}}}, C#108289
        {{2, 3}, {{{1, 2}, {1, 2}}}},
};

std::vector<ov::test::scatterNDUpdateSpecParams> combineShapes(const InputMap& input_shapes) {
    std::vector<ov::test::scatterNDUpdateSpecParams> resVec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            auto indices_shape = item.first;
            size_t indices_rank = indices_shape.size();
            std::vector<size_t> update_shape;
            for (size_t i = 0; i < indices_rank - 1; i++) {
                update_shape.push_back(indices_shape[i]);
            }
            auto src_shape = input_shape.first;
            for (size_t j = indices_shape[indices_rank - 1]; j < src_shape.size(); j++) {
                update_shape.push_back(src_shape[j]);
            }
            std::vector<ov::Shape> in_shapes{src_shape, update_shape};
            resVec.push_back(ov::test::scatterNDUpdateSpecParams{
                    ov::test::static_shapes_to_test_representation(in_shapes), ov::Shape{indices_shape}, item.second});
        }
    }
    return resVec;
}

const auto params = testing::Combine(testing::ValuesIn(combineShapes(sliceSelectInShape)),
                                     testing::Values(ov::element::f16),  // model
                                     testing::Values(ov::element::i32),  // indices
                                     testing::Values(test_utils::TARGET_DEVICE));

const auto precommit_params = testing::Combine(testing::ValuesIn(combineShapes(precommit_sliceSelectInShape)),
                                               testing::Values(ov::element::f16),  // model
                                               testing::Values(ov::element::i32),  // indices
                                               testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate, ScatterNDUpdateLayerTestMTLHW, params,
                         ScatterNDUpdateLayerTestMTLHW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ScatterNDUpdate, ScatterNDUpdateLayerTestCommon, precommit_params,
                         ScatterNDUpdateLayerTestCommon::getTestCaseName);

}  // namespace
