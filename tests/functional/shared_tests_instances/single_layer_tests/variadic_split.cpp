// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/variadic_split.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
    class KmbVariadicSplitLayerTest : public VariadicSplitLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    };

   TEST_P(KmbVariadicSplitLayerTest, CompareWithRefs_MLIR) {
       useCompilerMLIR();
       Run();
   }

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::SizeVector> inputShapes = {InferenceEngine::SizeVector{1, 144, 30, 40}};

    const InferenceEngine::Precision netPrecisions = InferenceEngine::Precision::FP32;

    const std::vector<size_t> numSplits = {64, 48, 32};

    INSTANTIATE_TEST_CASE_P(smoke_VariadicSplit, KmbVariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(numSplits),                                    //numSplits
                                           ::testing::Values(1),                                            //axis
                                           ::testing::Values(netPrecisions),                                //netPrecision
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      //inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      //outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),                 //inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),                 //outLayout
                                           ::testing::ValuesIn(inputShapes),                                //inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbVariadicSplitLayerTest::getTestCaseName);

    /* ============= Negative Axis ============= */

    INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis, KmbVariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),                    //numSplits
                                           ::testing::Values(-1),                                           //axis
                                           ::testing::Values(netPrecisions),                                //netPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      //inPrc
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),      //outPrc
                                           ::testing::Values(InferenceEngine::Layout::ANY),                 //inLayout
                                           ::testing::Values(InferenceEngine::Layout::ANY),                 //outLayout
                                           ::testing::Values(InferenceEngine::SizeVector{1, 384, 2}),       //inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbVariadicSplitLayerTest::getTestCaseName);
}  // namespace
