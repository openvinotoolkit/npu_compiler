// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/group_convolution_backprop_data.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GroupConvBackpropLayerTestCommon : public GroupConvBackpropLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(GroupConvBackpropLayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.1;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupConvBackpropLayerTestCommon, NPU4000_HW) {
    abs_threshold = 0.1;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(GroupConvBackpropLayerTestCommon, NPU5010_HW) {
    abs_threshold = 0.1;
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<size_t> numOutChannels = {64};
const std::vector<size_t> numGroups = {64};
const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= 2D GroupConvBackpropDataOp ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2D = {{{1, 64, 64, 64}}};
const std::vector<std::vector<size_t>> kernels2D = {{4, 4}};
const std::vector<std::vector<size_t>> strides2D = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> outputPadding2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding));

const auto groupConvBackpropData2DParams_OutputPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding2D));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropLayerTestCommon,
                         ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         GroupConvBackpropLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_OutputPadding, GroupConvBackpropLayerTestCommon,
                         ::testing::Combine(groupConvBackpropData2DParams_OutputPadding,
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         GroupConvBackpropLayerTestCommon::getTestCaseName);

/* ============= 1D GroupConvBackpropDataOp ============= */
const std::vector<std::vector<ov::Shape>> inputShapes1D = {{{1, 16, 64}}};
const std::vector<std::vector<size_t>> kernels1D = {{5}};
const std::vector<std::vector<size_t>> strides1D = {{2}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{1}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{1}};
const std::vector<std::vector<size_t>> dilations1D = {{1}};
const std::vector<std::vector<ptrdiff_t>> outputPadding1D = {{1}};
const std::vector<size_t> numOutChannels1D = {16};
const std::vector<size_t> numGroups1D = {16};

const auto groupConvBackpropData1DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D), ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D), ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
        ::testing::ValuesIn(numGroups1D), ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding));

const auto groupConvBackpropData1DParams_OutputPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels1D), ::testing::ValuesIn(strides1D), ::testing::ValuesIn(padBegins1D),
        ::testing::ValuesIn(padEnds1D), ::testing::ValuesIn(dilations1D), ::testing::ValuesIn(numOutChannels1D),
        ::testing::ValuesIn(numGroups1D), ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding1D));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData1D_ExplicitPadding, GroupConvBackpropLayerTestCommon,
                         ::testing::Combine(groupConvBackpropData1DParams_ExplicitPadding,
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes1D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         GroupConvBackpropLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData1D_OutputPadding, GroupConvBackpropLayerTestCommon,
                         ::testing::Combine(groupConvBackpropData1DParams_OutputPadding,
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes1D)),
                                            ::testing::ValuesIn(emptyOutputShape),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         GroupConvBackpropLayerTestCommon::getTestCaseName);

}  // namespace
