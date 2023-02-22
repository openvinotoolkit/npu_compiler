//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSplitLayerTest : public SplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
    }

    void SkipBeforeInfer() override {
        throw LayerTestsUtils::KmbSkipTestException(
                "Issues with Runtime. Outputs is empty because runtime doesn't wait while dma is finished");
    }
};
class KmbSplitLayerTest_MLIR_VPU3720 : public SplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbSplitLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbSplitLayerTest_MLIR_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        InferenceEngine::Precision::FP16   // tests: KmbGRNLayerTest, KmbSplitLayerTest, KmbCTCGreedyDecoderLayerTest
};

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Split, KmbSplitLayerTest,
        ::testing::Combine(::testing::Values(2, 3), ::testing::Values(0, 1, 2, 3), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                           ::testing::Values(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32),
                           ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                           ::testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
                           ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                           ::testing::Values(InferenceEngine::SizeVector({})),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        SplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_Split, KmbSplitLayerTest_MLIR_VPU3720,
                         ::testing::Combine(::testing::Values(2), ::testing::Values(1),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::SizeVector({6, 6, 12, 24})),
                                            ::testing::Values(InferenceEngine::SizeVector({})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         SplitLayerTest::getTestCaseName);

}  // namespace
