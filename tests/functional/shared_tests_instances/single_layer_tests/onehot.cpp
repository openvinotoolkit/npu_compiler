//
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/one_hot.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class OneHotLayerTestCommon : public OneHotLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(OneHotLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(OneHotLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
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

auto oneHotparams = [](auto onOffType) {
    return ::testing::Combine(::testing::Values(ov::element::i64), ::testing::ValuesIn(depthVal),
                              ::testing::Values(onOffType), ::testing::ValuesIn(onVal), ::testing::ValuesIn(offVal),
                              ::testing::ValuesIn(axis), ::testing::Values(ov::element::i32),
                              ::testing::ValuesIn(static_shapes_to_test_representation(inputShape)),
                              ::testing::Values(test_utils::TARGET_DEVICE));
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_OneHot_FP16, OneHotLayerTestCommon, oneHotparams(ov::element::f16),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_FP32, OneHotLayerTestCommon, oneHotparams(ov::element::f32),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_I32, OneHotLayerTestCommon, oneHotparams(ov::element::i32),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_I8, OneHotLayerTestCommon, oneHotparams(ov::element::i8),
                         OneHotLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OneHot_U8, OneHotLayerTestCommon, oneHotparams(ov::element::u8),
                         OneHotLayerTest::getTestCaseName);

}  // namespace
