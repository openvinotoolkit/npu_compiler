// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lstm_cell.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LSTMCellLayerTestCommon : public LSTMCellTest, virtual public VpuOv2LayerTest {};

TEST_P(LSTMCellLayerTestCommon, NPU3720_HW) {
    rel_threshold = 0.06;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LSTMCellLayerTestCommon, NPU4000_HW) {
    rel_threshold = 0.06;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
std::vector<bool> should_decompose{false};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<ov::element::Type> modelTypes = {ov::element::f16};

std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{4, 64};
std::vector<size_t> input_size{6, 24};
std::vector<size_t> hidden_size_precommit{2, 16};
std::vector<size_t> input_size_precommit{3, 12};

const auto lstmCellConfig = ::testing::Combine(
        ::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size),
        ::testing::ValuesIn(input_size), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto lstmCellPrecommitConfig = ::testing::Combine(
        ::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size_precommit),
        ::testing::ValuesIn(input_size_precommit), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LSTMCellCommon, LSTMCellLayerTestCommon, lstmCellPrecommitConfig,
                         LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMCellCommon, LSTMCellLayerTestCommon, lstmCellConfig, LSTMCellTest::getTestCaseName);

}  // namespace
