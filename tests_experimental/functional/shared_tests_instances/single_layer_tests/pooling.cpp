// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/pooling.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    //         InferenceEngine::Precision::FP16, // "[NOT_IMPLEMENTED] Input image format FP16 is not supported yet...
    InferenceEngine::Precision::U8,
    //         InferenceEngine::Precision::I8 // Too much cases
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3}, {3, 5}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1}, {1, 2}};
const std::vector<InferenceEngine::SizeVector> padBegins = {{0, 0}, {0, 2}};
const std::vector<InferenceEngine::SizeVector> padEnds = {{0, 0}, {0, 2}};
const std::vector<ngraph::op::RoundingType> roundingTypes = {
    ngraph::op::RoundingType::CEIL, ngraph::op::RoundingType::FLOOR};
////* ========== Max Polling ========== */
/* +========== Explicit Pad Floor Rounding ========== */
const auto maxPool_ExplicitPad_FloorRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(MaxPool_ExplicitPad_FloorRpunding, PoolingLayerTest,
    ::testing::Combine(maxPool_ExplicitPad_FloorRounding_Params, ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    PoolingLayerTest::getTestCaseName);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params =
    ::testing::Combine(::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at
        // axis 3" thrown in the test body.
        ::testing::Values(InferenceEngine::SizeVector({1, 1})), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(MaxPool_ExplicitPad_CeilRpunding, PoolingLayerTest,
    ::testing::Combine(maxPool_ExplicitPad_CeilRounding_Params, ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg Pooling ========== */
/* +========== Explicit Pad Ceil Rounding ========== */
const auto avgPoolExplicitPadCeilRoundingParams =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG), ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at
        // axis 3" thrown in the test body.
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        // TODO: Non zero pads excluded because of accuracy mismatch
        ::testing::Values(InferenceEngine::SizeVector({0, 0})), ::testing::Values(InferenceEngine::SizeVector({0, 0})),
        ::testing::Values(ngraph::op::RoundingType::CEIL), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true, false));

INSTANTIATE_TEST_CASE_P(AvgPool_ExplicitPad_CeilRounding, PoolingLayerTest,
    ::testing::Combine(avgPoolExplicitPadCeilRoundingParams, ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    PoolingLayerTest::getTestCaseName);

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolExplicitPadFloorRoundingParams =
    ::testing::Combine(::testing::Values(PoolingTypes::AVG), ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
        // TODO: Non zero pads excluded because of accuracy mismatch
        ::testing::Values(InferenceEngine::SizeVector({0, 0})), ::testing::Values(InferenceEngine::SizeVector({0, 0})),
        ::testing::Values(ngraph::op::RoundingType::FLOOR), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true, false));

INSTANTIATE_TEST_CASE_P(AvgPool_ExplicitPad_FloorRounding, PoolingLayerTest,
    ::testing::Combine(avgPoolExplicitPadFloorRoundingParams, ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Polling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::Values(InferenceEngine::SizeVector({0, 0})),
    ::testing::Values(InferenceEngine::SizeVector({0, 0})),
    ::testing::Values(
        ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
    ::testing::Values(ngraph::op::PadType::VALID),
    ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(MAX_and_AVGPool_ValidPad, PoolingLayerTest,
    ::testing::Combine(allPools_ValidPad_Params, ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
    PoolingLayerTest::getTestCaseName);

}  // namespace
