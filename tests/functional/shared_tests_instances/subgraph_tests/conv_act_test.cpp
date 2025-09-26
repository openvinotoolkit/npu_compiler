// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/conv_act_base.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {
namespace test {

using namespace ov::test::utils;

class ConvActivationSubgraphTestCommon : public ConvActTest, virtual public VpuOv2LayerTest {};
class ConvActivationSubgraphTest_NPU3720 : public ConvActivationSubgraphTestCommon {};
class ConvActivationSubgraphTest_NPU4000 : public ConvActivationSubgraphTestCommon {};

}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {

auto static_shapes_param_transform =
        [](const std::vector<std::pair<std::vector<ov::Shape>, ov::Shape>>& original_shapes) {
            std::vector<std::pair<std::vector<InputShape>, ov::Shape>> new_shapes;
            for (const auto& shape_element : original_shapes) {
                new_shapes.emplace_back(static_shapes_to_test_representation(shape_element.first),
                                        shape_element.second);
            }
            return new_shapes;
        };

const std::vector<ov::element::Type> modelTypes = {
        ov::element::u8,
};

/* ============= 2D Convolution ============= */

const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{1, 1}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};

/* ============= 3D Convolution ============= */

const std::vector<std::vector<size_t>> kernels3D = {{3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}};

const std::vector<size_t> numOutCannels = {64};
const std::vector<ov::op::PadType> padTypes = {ov::op::PadType::EXPLICIT, ov::op::PadType::VALID};

const std::vector<ov::element::Type> inputPrecisions = {ov::element::f32};

const std::vector<ov::element::Type> outputPrecisions = {ov::element::f16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Mish, {{}}},
        {LeakyRelu, {{0.1f}}},
};

std::map<std::vector<ov::Shape>, std::vector<ov::Shape>> basic = {
        {{{1, 64, 1, 1}}, {{}}},
        //{{{1, 50, 1, 1}}, {{}}}, - error in fp16 network
        {{{1, 128, 1, 1}}, {{}}},  // should cover most of u8 values
};

const auto activationCases =
        ::testing::Combine(::testing::ValuesIn(combineParams(activationTypes)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_param_transform(combineParams(basic))),
                           ::testing::Values(test_utils::TARGET_DEVICE));

const auto convCases =
        ::testing::Combine(activationCases, ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
                           ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
                           ::testing::ValuesIn(numOutCannels), ::testing::Values(ov::op::PadType::EXPLICIT));

}  // namespace
