// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/deformable_convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class DeformableConvolutionLayerTestCommon : public DeformableConvolutionLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(DeformableConvolutionLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(DeformableConvolutionLayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<bool> bilinearInterpolatePad = {true, false};

const auto configParamsStrides1x1 =
        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),     // strides
                           ::testing::Values(std::vector<ptrdiff_t>{1, 1}),  // pad begin
                           ::testing::Values(std::vector<ptrdiff_t>{1, 1}),  // pad end
                           ::testing::Values(std::vector<size_t>{1, 1}),     // dilation
                           ::testing::Values(1),                             // group
                           ::testing::Values(1),                             // deformable group
                           ::testing::Values(4),                             // num out channels
                           ::testing::Values(ov::op::PadType::EXPLICIT),     // pad type
                           ::testing::ValuesIn(bilinearInterpolatePad));     // bilinear interpolate pad

const auto testParamsStrides1x1 =
        ::testing::Combine(configParamsStrides1x1,               // def conv params
                           ::testing::Values(true),              // modulation
                           ::testing::Values(ov::element::f16),  // model type
                           ::testing::Values(ov::test::static_shapes_to_test_representation(
                                   {{1, 32, 19, 19}, {1, 18, 19, 19}, {32, 32, 3, 3}, {1, 9, 19, 19}})),  // input shape
                           ::testing::Values(test_utils::TARGET_DEVICE));                                 // device name

const auto configParamsStrides2x2 =
        ::testing::Combine(::testing::Values(std::vector<size_t>{2, 2}),     // strides
                           ::testing::Values(std::vector<ptrdiff_t>{1, 1}),  // pad begin
                           ::testing::Values(std::vector<ptrdiff_t>{1, 1}),  // pad end
                           ::testing::Values(std::vector<size_t>{1, 1}),     // dilation
                           ::testing::Values(1),                             // group
                           ::testing::Values(1),                             // deformable group
                           ::testing::Values(4),                             // num out channels
                           ::testing::Values(ov::op::PadType::EXPLICIT),     // pad type
                           ::testing::ValuesIn(bilinearInterpolatePad));     // bilinear interpolate pad

const auto testParamsStrides2x2 =
        ::testing::Combine(configParamsStrides2x2,               // def conv params
                           ::testing::Values(true),              // modulation
                           ::testing::Values(ov::element::f16),  // model type
                           ::testing::Values(ov::test::static_shapes_to_test_representation(
                                   {{1, 32, 38, 38}, {1, 18, 19, 19}, {32, 32, 3, 3}, {1, 9, 19, 19}})),  // input shape
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto configParamsMultipleDG =
        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),     // strides
                           ::testing::Values(std::vector<ptrdiff_t>{0, 0}),  // pad begin
                           ::testing::Values(std::vector<ptrdiff_t>{0, 0}),  // pad end
                           ::testing::Values(std::vector<size_t>{1, 1}),     // dilation
                           ::testing::Values(1),                             // group
                           ::testing::Values(2),                             // deformable group
                           ::testing::Values(1),                             // num out channels
                           ::testing::Values(ov::op::PadType::EXPLICIT),     // pad type
                           ::testing::ValuesIn(bilinearInterpolatePad));     // bilinear interpolate pad

const auto testParamsMultipleDG =
        ::testing::Combine(configParamsMultipleDG,               // def conv params
                           ::testing::Values(true),              // modulation
                           ::testing::Values(ov::element::f16),  // model type
                           ::testing::Values(ov::test::static_shapes_to_test_representation(
                                   {{1, 2, 3, 3}, {1, 16, 2, 2}, {2, 2, 2, 2}, {1, 8, 2, 2}})),  // input shape
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precomit_DeformableConvolution2DTest_Strides1x1, DeformableConvolutionLayerTestCommon,
                         testParamsStrides1x1, DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DeformableConvolution2DTest_Strides2x2, DeformableConvolutionLayerTestCommon,
                         testParamsStrides2x2, DeformableConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precomit_DeformableConvolution_MultipleDG, DeformableConvolutionLayerTestCommon,
                         testParamsMultipleDG, DeformableConvolutionLayerTest::getTestCaseName);

}  // namespace
