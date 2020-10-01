// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/softmax.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSoftMaxLayerTest: public SoftMaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

// TODO: [Track number: C#38227]
TEST_P(KmbSoftMaxLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inputShapes2D = {
    InferenceEngine::SizeVector {1, 100},
    InferenceEngine::SizeVector {100, 1},
    InferenceEngine::SizeVector {10, 10},
};

const std::vector<size_t> axis2D = {
    0, 1
};

const auto params2D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::ValuesIn(inputLayouts2D),
    testing::ValuesIn(inputShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    smoke_SoftMax2D,
    KmbSoftMaxLayerTest,
    params2D,
    SoftMaxLayerTest::getTestCaseName
);

const std::vector<InferenceEngine::SizeVector> inputShapes4D = {
    InferenceEngine::SizeVector {1, 100, 1, 1},
    InferenceEngine::SizeVector {1, 3, 4, 3},
    InferenceEngine::SizeVector {2, 3, 4, 5},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::ValuesIn(inputShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    smoke_SoftMax4D,
    KmbSoftMaxLayerTest,
    params4D,
    SoftMaxLayerTest::getTestCaseName
);

}  // namespace
