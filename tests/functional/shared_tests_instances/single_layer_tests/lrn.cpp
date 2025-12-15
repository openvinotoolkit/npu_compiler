//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lrn.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LrnLayerTestCommon_FP16 : public LrnLayerTest, virtual public VpuOv2LayerTest {};
class LrnLayerTestCommon_FP32 : public LrnLayerTestCommon_FP16 {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(LrnLayerTestCommon_FP16, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LrnLayerTestCommon_FP16, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(LrnLayerTestCommon_FP32, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LrnLayerTestCommon_FP32, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(LrnLayerTestCommon_FP16, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(LrnLayerTestCommon_FP32, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

const std::vector<std::vector<int64_t>> axes = {{1}, {2}, {1, 2}, {2, 3}, {1, 2, 3}};

const auto lrnParams_FP16 = ::testing::Combine(
        ::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias), ::testing::Values(size),
        ::testing::ValuesIn(axes), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 3, 2}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto lrnGooglenetV1Params_FP16 = ::testing::Combine(
        ::testing::Values(9.9e-05),                    // alpha
        ::testing::Values(0.75),                       // beta
        ::testing::Values(1.0),                        // bias
        ::testing::Values(5),                          // size
        ::testing::Values(std::vector<int64_t>({1})),  // axes
        ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 64, 56, 56}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

const auto lrnParams_FP32 = ::testing::Combine(::testing::Values(9.9e-05),                    // alpha
                                               ::testing::Values(0.75),                       // beta
                                               ::testing::Values(1.0),                        // bias
                                               ::testing::Values(5),                          // size
                                               ::testing::Values(std::vector<int64_t>({1})),  // axes
                                               ::testing::Values(ov::element::f32),
                                               ::testing::ValuesIn(static_shapes_to_test_representation(
                                                       std::vector<std::vector<ov::Shape>>({{{1, 32, 56, 56}}}))),
                                               ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnCheck, LrnLayerTestCommon_FP16, lrnParams_FP16,
                         LrnLayerTestCommon_FP16::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnGooglenetV1, LrnLayerTestCommon_FP16, lrnGooglenetV1Params_FP16,
                         LrnLayerTestCommon_FP16::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnGooglenetV1_FP32, LrnLayerTestCommon_FP32, lrnParams_FP32,
                         LrnLayerTestCommon_FP32::getTestCaseName);

}  // namespace
