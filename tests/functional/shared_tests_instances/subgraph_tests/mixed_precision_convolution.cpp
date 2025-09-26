// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mixed_precision_convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

class MixedPrecisionConvSubGraphTestCommon : public MixedPrecisionConvSubGraphTest {};

using MixedPrecisionConvSubGraphTest_NPU3720 = MixedPrecisionConvSubGraphTestCommon;
using MixedPrecisionConvSubGraphTest_NPU4000 = MixedPrecisionConvSubGraphTestCommon;

TEST_P(MixedPrecisionConvSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MixedPrecisionConvSubGraphTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
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

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I8, MixedPrecisionConvSubGraphTest_NPU3720,
                         ::testing::Combine(conv2DParamsI8,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I4, MixedPrecisionConvSubGraphTest_NPU3720,
                         ::testing::Combine(conv2DParamsI4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I8, MixedPrecisionConvSubGraphTest_NPU4000,
                         ::testing::Combine(conv2DParamsI8,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_I4, MixedPrecisionConvSubGraphTest_NPU4000,
                         ::testing::Combine(conv2DParamsI4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_mixed_precision_Convolution2D_NF4, MixedPrecisionConvSubGraphTest_NPU4000,
                         ::testing::Combine(conv2DParamsNF4,
                                            ::testing::Values(ov::element::f16),              // netPrc
                                            ::testing::ValuesIn({ov::Shape{1, 16, 16, 16}}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),    // targetDevice
                         MixedPrecisionConvSubGraphTest_NPU4000::getTestCaseName);

}  // namespace
