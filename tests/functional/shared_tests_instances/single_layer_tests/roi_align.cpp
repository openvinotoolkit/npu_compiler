// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/roi_align.hpp"

namespace LayerTestsDefinitions {

    class KmbROIAlignLayerTest: public ROIAlignLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

    };

    TEST_P(KmbROIAlignLayerTest, CompareWithRefs_MLIR) {
        useCompilerMLIR();
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecision = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::vector<size_t>> inputShape = {
            { 2, 18, 20, 20 },
            { 2, 4, 20, 20 },
            { 2, 4, 20, 40 },
            { 10, 1, 20, 20 }
    };

    const std::vector<std::vector<unsigned long>> coordsShape = {
            {2, 4}
    };

    const std::vector<int> pooledH = { 2 };

    const std::vector<int> pooledW = { 2 };

    const std::vector<float> spatialScale = { 0.625f, 1.0f };

    const std::vector<int> poolingRatio = { 2 };

    const std::vector<std::string> poolingMode = {
            "avg",
            "max"
    };


    const auto roialignparams = testing::Combine(
            testing::ValuesIn(inputShape),
            testing::ValuesIn(coordsShape),
            testing::ValuesIn(pooledH),
            testing::ValuesIn(pooledW),
            testing::ValuesIn(spatialScale),
            testing::ValuesIn(poolingRatio),
            testing::ValuesIn(poolingMode),
            testing::ValuesIn(netPrecision),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
            );

    INSTANTIATE_TEST_CASE_P(
            smoke_ROIAlign,
            KmbROIAlignLayerTest,
            roialignparams,
            KmbROIAlignLayerTest::getTestCaseName
    );
}  // namespace