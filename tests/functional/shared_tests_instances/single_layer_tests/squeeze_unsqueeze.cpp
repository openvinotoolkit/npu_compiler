//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/squeeze_unsqueeze.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {
class SqueezeUnsqueezeLayerTestCommon : public SqueezeUnsqueezeLayerTest, virtual public VpuOv2LayerTest {
protected:
    ov::test::utils::SkipCallback skipCompilationCallback = [this](std::stringstream& str) {
        const auto inRank = function->get_parameters().at(0)->get_output_shape(0).size();
        const auto outRank = function->get_results().at(0)->get_input_shape(0).size();
        if (inRank > 4 || outRank > 4) {
            str << "> 4D case is not supported";
        }
    };
};

TEST_P(SqueezeUnsqueezeLayerTestCommon, NPU3720_HW) {
    setSkipCompilationCallback(skipCompilationCallback);
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SqueezeUnsqueezeLayerTestCommon, NPU4000_HW) {
    setSkipCompilationCallback(skipCompilationCallback);
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(SqueezeUnsqueezeLayerTestCommon, NPU5010_HW) {
    setSkipCompilationCallback(skipCompilationCallback);
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using ov::test::SqueezeUnsqueezeLayerTestCommon;

namespace {

std::map<std::vector<ov::Shape>, std::vector<std::vector<int>>> axesVectors = {
        {{{1, 1, 1, 1}},
         {{-1},
          {0},
          {1},
          {2},
          {3},
          {0, 1},
          {0, 2},
          {0, 3},
          {1, 2},
          {2, 3},
          {0, 1, 2},
          {0, 2, 3},
          {1, 2, 3},
          {0, 1, 2, 3}}},
        {{{1, 2, 3, 4}}, {{0}}},
        {{{2, 1, 3, 4}}, {{1}}},
        {{{1}}, {{-1}, {0}}},
        {{{1, 2}}, {{0}}},
        {{{2, 1}}, {{1}, {-1}}},
};

auto combined_axes = ov::test::utils::combineParams(axesVectors);

auto prepare_cases = [](const std::vector<std::pair<std::vector<ov::Shape>, std::vector<int>>>& raw_axes) {
    std::vector<std::pair<std::vector<ov::test::InputShape>, std::vector<int>>> cases;
    for (const auto& raw_case : raw_axes) {
        cases.emplace_back(ov::test::static_shapes_to_test_representation(raw_case.first), raw_case.second);
    }
    return cases;
};

auto axes = prepare_cases(combined_axes);

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<ov::test::utils::SqueezeOpType> opTypes = {ov::test::utils::SqueezeOpType::SQUEEZE,
                                                             ov::test::utils::SqueezeOpType::UNSQUEEZE};
const auto paramConfig =
        testing::Combine(::testing::ValuesIn(axes), ::testing::ValuesIn(opTypes), ::testing::ValuesIn(modelTypes),
                         ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Basic, SqueezeUnsqueezeLayerTestCommon, paramConfig,
                         SqueezeUnsqueezeLayerTestCommon::getTestCaseName);

}  // namespace
