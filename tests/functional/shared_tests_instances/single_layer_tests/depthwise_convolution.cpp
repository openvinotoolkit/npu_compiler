//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/group_convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GroupConvolutionLayerTest);

class DepthwiseConvolutionLayerTest_HW : public GroupConvolutionLayerTest, virtual public VpuOv2LayerTest {};

class DepthwiseConvolutionSCFTilingLayerTest_HW : public DepthwiseConvolutionLayerTest_HW {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "scf-tiling=true";
        // E-190336 for MC support
        configuration["NPU_TILES"] = "1";
    }
};

TEST_P(DepthwiseConvolutionLayerTest_HW, NPU4000) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DepthwiseConvolutionSCFTilingLayerTest_HW, NPU4000) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DepthwiseConvolutionLayerTest_HW, NPU5010) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DepthwiseConvolutionSCFTilingLayerTest_HW, NPU5010) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(DepthwiseConvolutionLayerTest_HW, NPU5020) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

/* ============= 2D DepthwiseConvolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {4, 4}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};

auto combineDepthwiseConv2D(size_t channels, size_t w = 30, size_t h = 30) {
    const auto parameters =
            ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
                               ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
                               ::testing::ValuesIn(dilations), /*numOutChannels*/ ::testing::Values(channels),
                               /*numGroups*/ ::testing::Values(channels), ::testing::Values(ov::op::PadType::VALID));

    return ::testing::Combine(parameters, ::testing::ValuesIn(modelTypes),
                              ::testing::ValuesIn({static_shapes_to_test_representation(
                                      {std::vector<ov::Shape>({{1, channels, w, h}})})}),
                              ::testing::Values(test_utils::TARGET_DEVICE));
}

INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_16_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(16), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_32_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(32), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_64_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(64), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_96_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(96), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_111_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(111), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_128_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(128), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_160_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(160), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_192_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(192), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_200_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(200), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_224_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(224), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_256_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(256), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_384_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(384), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_512_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(512), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_1024_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(1024), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_1111_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(1111), DepthwiseConvolutionLayerTest_HW::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_2048_channels, DepthwiseConvolutionLayerTest_HW,
                         combineDepthwiseConv2D(2048), DepthwiseConvolutionLayerTest_HW::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthwiseConvolution2D_64_channels_tiling, DepthwiseConvolutionSCFTilingLayerTest_HW,
                         combineDepthwiseConv2D(64, 300, 300),
                         DepthwiseConvolutionSCFTilingLayerTest_HW::getTestCaseName);

}  // namespace
