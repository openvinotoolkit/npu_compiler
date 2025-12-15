//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/group_convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

class GroupConvolutionLayerTestCommon : public GroupConvolutionLayerTest, virtual public VpuOv2LayerTest {};
class GroupConvolutionLayerTest_SW : public GroupConvolutionLayerTestCommon {};
class GroupConvolutionLayerTest_HW : public GroupConvolutionLayerTestCommon {};

TEST_P(GroupConvolutionLayerTest_HW, NPU3720) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupConvolutionLayerTest_SW, NPU3720) {
    rel_threshold = 0.01;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupConvolutionLayerTest_HW, NPU4000) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(GroupConvolutionLayerTest_SW, NPU4000) {
    rel_threshold = 0.01;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(GroupConvolutionLayerTest_HW, NPU5010) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(GroupConvolutionLayerTest_SW, NPU5010) {
    rel_threshold = 0.01;
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {
        ov::element::f16,
};

/* ============= 1D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels1d = {{3}};
const std::vector<std::vector<size_t>> strides1d = {{1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1d = {{0}, {2}};
const std::vector<std::vector<ptrdiff_t>> padEnds1d = {{0}, {2}};
const std::vector<std::vector<size_t>> dilations1d = {{1}, {2}};
const std::vector<size_t> numOutChannels1d = {8, 16};
const std::vector<size_t> numGroups1d = {2, 8};
const std::vector<std::vector<ov::Shape>> inputShapes1d = {{{1, 16, 30}}};

const auto groupConv1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d), ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d), ::testing::ValuesIn(dilations1d), ::testing::ValuesIn(numOutChannels1d),
        ::testing::ValuesIn(numGroups1d), ::testing::Values(ov::op::PadType::EXPLICIT));
const auto groupConv1DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d), ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<ptrdiff_t>({0})), ::testing::ValuesIn(dilations1d),
        ::testing::ValuesIn(numOutChannels1d), ::testing::ValuesIn(numGroups1d),
        ::testing::Values(ov::op::PadType::VALID));

const auto groupConv1D_ExplicitPadding =
        testing::Combine(groupConv1DParams_ExplicitPadding, ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes1d)),
                         ::testing::Values(test_utils::TARGET_DEVICE));
const auto groupConv1D_AutoPadValid =
        testing::Combine(groupConv1DParams_AutoPadValid, ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes1d)),
                         ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution1D_ExplicitPadding, GroupConvolutionLayerTest_SW,
                         groupConv1D_ExplicitPadding, GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution1D_AutoPadValid, GroupConvolutionLayerTest_SW, groupConv1D_AutoPadValid,
                         GroupConvolutionLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 32};
const std::vector<size_t> numGroups = {2, 8};
const std::vector<std::vector<ov::Shape>> inputShapes = {{{1, 32, 30, 30}}, {{1, 16, 30, 30}}};

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ov::op::PadType::EXPLICIT));
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels), ::testing::ValuesIn(numGroups), ::testing::Values(ov::op::PadType::VALID));

const auto groupConv2DParams_LargeStrides = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::Values(std::vector<size_t>({9, 9})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels), ::testing::ValuesIn(numGroups),
        ::testing::Values(ov::op::PadType::VALID));

const auto groupConv2D_ExplicitPadding =
        testing::Combine(groupConv2DParams_ExplicitPadding, ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn({static_shapes_to_test_representation({inputShapes[0]})}),
                         ::testing::Values(test_utils::TARGET_DEVICE));
const auto groupConv2D_AutoPadValid =
        testing::Combine(groupConv2DParams_AutoPadValid, ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn({static_shapes_to_test_representation({inputShapes[1]})}),
                         ::testing::Values(test_utils::TARGET_DEVICE));
const auto groupConv2D_LargeStrides =
        testing::Combine(groupConv2DParams_LargeStrides, ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn({static_shapes_to_test_representation({inputShapes[1]})}),
                         ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_ExplicitPadding, GroupConvolutionLayerTest_SW,
                         groupConv2D_ExplicitPadding, GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_AutoPadValid, GroupConvolutionLayerTest_SW, groupConv2D_AutoPadValid,
                         GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_LargeStrides, GroupConvolutionLayerTest_SW, groupConv2D_LargeStrides,
                         GroupConvolutionLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}};
const std::vector<std::vector<ov::Shape>> inputShapes3d = {{{1, 4, 10, 10, 10}}};

const auto groupConv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d), ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(dilations3d), ::testing::Values(4), ::testing::Values(2),
        ::testing::Values(ov::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D_ExplicitPadding, GroupConvolutionLayerTest_HW,
                         ::testing::Combine(groupConv3DParams_ExplicitPadding, ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes3d)),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
