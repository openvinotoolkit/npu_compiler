//
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/gru_cell.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GRUCellLayerTestCommon : public GRUCellTest, virtual public VpuOv2LayerTest {};
class GRUCellLayerTest_NPU3720 : public GRUCellLayerTestCommon {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        GRUCellTest::SetUp();
    }
};

TEST_P(GRUCellLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GRUCellLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(GRUCellLayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<bool> shouldDecompose{false};
const std::vector<size_t> batch{2};
const std::vector<size_t> hiddenSize{128};
const std::vector<size_t> inputSize{128};
const std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
const std::vector<float> clip{0.f};
const std::vector<bool> shouldLinearBeforeReset{true, false};
const std::vector<ov::element::Type> modelTypes = {ov::element::f16};
const std::vector<InputLayerType> WRBLayerTypes = {InputLayerType::CONSTANT};

const auto gruCellParams = testing::Combine(
        ::testing::ValuesIn(shouldDecompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hiddenSize),
        ::testing::ValuesIn(inputSize), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::ValuesIn(shouldLinearBeforeReset), ::testing::ValuesIn(WRBLayerTypes),
        ::testing::ValuesIn(WRBLayerTypes), ::testing::ValuesIn(WRBLayerTypes), ::testing::ValuesIn(modelTypes),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUCell, GRUCellLayerTest_NPU3720, gruCellParams,
                         GRUCellTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUCell, GRUCellLayerTestCommon, gruCellParams, GRUCellTest::getTestCaseName);

}  // namespace
