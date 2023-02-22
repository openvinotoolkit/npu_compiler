//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/bucketize.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbBucketizeLayerTest : public BucketizeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        std::tie(std::ignore /*Data shape*/, std::ignore /*Buckets shape*/, std::ignore /*Right edge of interval*/,
                 std::ignore /*Data input precision*/, std::ignore /*Buckets input precision*/, outPrc,
                 std::ignore /*deviceName*/
                 ) = GetParam();

        BucketizeLayerTest::SetUp();
    }

    void SkipBeforeLoad() override {
        if (outPrc == InferenceEngine::Precision::I64) {
            throw LayerTestsUtils::KmbSkipTestException("I64 Precision is not supported yet!");
        }
    }
};

TEST_P(KmbBucketizeLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> dataInputPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> bucketsInputPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> outputPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64,  // Skipped before load
};

const std::vector<std::vector<size_t>> dataShapes = {{1, 20, 20}, {2, 3, 50, 50}};

const std::vector<std::vector<size_t>> bucketsShapes = {{100}};

const std::vector<bool> with_right_bound = {true, false};

const auto testBucketizeParams = ::testing::Combine(
        ::testing::ValuesIn(dataShapes), ::testing::ValuesIn(bucketsShapes), ::testing::ValuesIn(with_right_bound),
        ::testing::ValuesIn(dataInputPrecisions), ::testing::ValuesIn(bucketsInputPrecisions),
        ::testing::Values(outputPrecisions[0]), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_BucketizeTest, KmbBucketizeLayerTest, testBucketizeParams,
                        KmbBucketizeLayerTest::getTestCaseName);

const auto testBucketizeParamsI64 = ::testing::Combine(
        ::testing::Values(dataShapes[0]), ::testing::ValuesIn(bucketsShapes), ::testing::ValuesIn(with_right_bound),
        ::testing::Values(dataInputPrecisions[0]), ::testing::Values(bucketsInputPrecisions[0]),
        ::testing::Values(outputPrecisions[1]), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_BucketizeTestI64, KmbBucketizeLayerTest, testBucketizeParamsI64,
                        KmbBucketizeLayerTest::getTestCaseName);

}  // namespace
