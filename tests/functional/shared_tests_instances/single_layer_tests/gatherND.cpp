// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <vpux/vpux_plugin_config.hpp>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/gather_nd.hpp"

namespace LayerTestsDefinitions {

class KmbGatherNDLayerTest : public GatherND8LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class KmbGatherNDLayerTest_VPU3720 : public GatherND8LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbGatherNDLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbGatherNDLayerTest_VPU3720, SW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> dPrecisions = {
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> iPrecisions = {
        InferenceEngine::Precision::I32,
};

const auto gatherNDArgsSubset1 = testing::Combine(
        testing::Combine(testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 2}, {2, 3, 4}})),  // Data shape
                         testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 1}, {2, 1, 1}})),  // Indices shape
                         testing::ValuesIn(std::vector<int>({0, 1}))),                              // Batch dims
        testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice), testing::Values(Config{}));

const auto gatherNDArgsSubset2 =
        testing::Combine(testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 3, 4, 3, 17}})),
                                          ::testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 3, 2, 3}})),
                                          ::testing::ValuesIn(std::vector<int>({0, 1, 2}))),
                         testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice), testing::Values(Config{}));

const auto gatherNDArgsSubsetPrecommit = testing::Combine(
        testing::Combine(testing::ValuesIn(std::vector<std::vector<size_t>>({{5, 7, 3}})),
                         testing::ValuesIn(std::vector<std::vector<size_t>>({{5, 1}})),
                         testing::ValuesIn(std::vector<int>({1}))),
        testing::Values(InferenceEngine::Precision::I32), testing::Values(InferenceEngine::Precision::I32),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice), testing::Values(Config{}));

const auto gatherNDArgsSubsetTiling = testing::Combine(
        testing::Combine(testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 5, 128, 512}})),
                         testing::ValuesIn(std::vector<std::vector<size_t>>({{2, 1, 100, 2}})),
                         testing::ValuesIn(std::vector<int>({1}))),
        testing::Values(InferenceEngine::Precision::I32), testing::Values(InferenceEngine::Precision::I32),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice), testing::Values(Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_GatherND_Set1, KmbGatherNDLayerTest, gatherNDArgsSubset1,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND_Set2, KmbGatherNDLayerTest, gatherNDArgsSubset2,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherND_VPU3720, KmbGatherNDLayerTest_VPU3720, gatherNDArgsSubset1,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherND_VPU3720, KmbGatherNDLayerTest_VPU3720, gatherNDArgsSubsetPrecommit,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_GatherND_VPU3720, KmbGatherNDLayerTest_VPU3720, gatherNDArgsSubsetTiling,
                         GatherND8LayerTest::getTestCaseName);

}  // namespace
