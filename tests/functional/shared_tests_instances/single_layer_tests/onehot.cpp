//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/one_hot.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OneHot1LayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OneHot16LayerTest);

class OneHot16LayerTestCommon : public OneHot16LayerTest, virtual public VpuOv2LayerTest {};

TEST_P(OneHot16LayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(OneHot16LayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(OneHot16LayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(OneHot16LayerTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<int64_t> depthVal{3};
const std::vector<float> onVal{1.0f};
const std::vector<float> offVal{0.0f};
const std::vector<int64_t> axis{-2, 0};
const std::vector<std::vector<ov::Shape>> inputShape = {
        std::vector<ov::Shape>{{4}},
        std::vector<ov::Shape>{{2, 3}},
};

const std::vector<ov::op::v16::OneHot::NegativeIndicesMode> neg_ind_mode = {
        ov::op::v16::OneHot::NegativeIndicesMode::IGNORE_NEGATIVE, ov::op::v16::OneHot::NegativeIndicesMode::NORMALIZE};

auto oneHotparams = [](auto onOffType) {
    return ::testing::Combine(::testing::Values(ov::element::i64), ::testing::ValuesIn(depthVal),
                              ::testing::Values(onOffType), ::testing::ValuesIn(onVal), ::testing::ValuesIn(offVal),
                              ::testing::ValuesIn(axis), ::testing::Values(ov::element::i32),
                              ::testing::ValuesIn(static_shapes_to_test_representation(inputShape)),
                              ::testing::ValuesIn(neg_ind_mode), ::testing::Values(test_utils::TARGET_DEVICE));
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_OneHot_FP16, OneHot16LayerTestCommon, oneHotparams(ov::element::f16),
                         OneHot16LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_FP32, OneHot16LayerTestCommon, oneHotparams(ov::element::f32),
                         OneHot16LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_I32, OneHot16LayerTestCommon, oneHotparams(ov::element::i32),
                         OneHot16LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_I8, OneHot16LayerTestCommon, oneHotparams(ov::element::i8),
                         OneHot16LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_U8, OneHot16LayerTestCommon, oneHotparams(ov::element::u8),
                         OneHot16LayerTest::getTestCaseName);

}  // namespace
