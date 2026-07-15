//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mixed_precision_convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class MixedPrecisionConvSubGraphTestCommon : public MixedPrecisionConvSubGraphTest {};

TEST_P(MixedPrecisionConvSubGraphTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MixedPrecisionConvSubGraphTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

using MixedPrecisionConvSubGraphTestNF4 = MixedPrecisionConvSubGraphTestCommon;
TEST_P(MixedPrecisionConvSubGraphTestNF4, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(MixedPrecisionConvSubGraphTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(MixedPrecisionConvSubGraphTestNF4, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
using namespace ov::test::utils;

namespace {

const auto conv2DParamsI8 =
        ::testing::Combine(::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padEnds
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // dilations
                           ::testing::Values(16),                                    // numOutChannels
                           ::testing::Values(LowFpType::Undefined),                  // lowFpType
                           ::testing::Values(255),                                   // quantLevels
                           ::testing::Values(QuantizationGranularity::Pertensor)     // quantGranularity
        );

const auto conv2DParamsI4 =
        ::testing::Combine(::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padEnds
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // dilations
                           ::testing::Values(16),                                    // numOutChannels
                           ::testing::Values(LowFpType::Undefined),                  // lowFpType
                           ::testing::Values(16),                                    // quantLevels
                           ::testing::Values(QuantizationGranularity::Pertensor)     // quantGranularity
        );

const auto conv2DParamsNF4 =
        ::testing::Combine(::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),    // padEnds
                           ::testing::ValuesIn<std::vector<std::size_t>>({{1, 1}}),  // dilations
                           ::testing::Values(16),                                    // numOutChannels
                           ::testing::Values(LowFpType::NF4),                        // lowFpType
                           ::testing::Values(0),                                     // quantLevels
                           ::testing::Values(QuantizationGranularity::Pertensor)     // quantGranularity
        );

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I8, MixedPrecisionConvSubGraphTestCommon,
                         ::testing::Combine(conv2DParamsI8,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I4, MixedPrecisionConvSubGraphTestCommon,
                         ::testing::Combine(conv2DParamsI4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTestCommon::getTestCaseName);

// nf4 test cases (NPU4000+)
INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_NF4, MixedPrecisionConvSubGraphTestNF4,
                         ::testing::Combine(conv2DParamsNF4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTestNF4::getTestCaseName);

}  // namespace
