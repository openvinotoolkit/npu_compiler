//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/roi_pooling.hpp"

namespace LayerTestsDefinitions {

class KmbROIPoolingLayerTest : public ROIPoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void GenerateInputs() override {
        ngraph::helpers::ROIPoolingTypes poolMethod;
        float spatialScale = 0.f;
        std::tie(std::ignore, std::ignore, std::ignore, spatialScale, poolMethod, std::ignore, std::ignore) =
                GetParam();

        inputs.clear();

        auto feat_map_shape = cnnNetwork.getInputShapes().begin()->second;

        const auto is_roi_max_mode = (poolMethod == ngraph::helpers::ROIPoolingTypes::ROI_MAX);

        const int height = is_roi_max_mode ? feat_map_shape[2] / spatialScale : 1;
        const int width = is_roi_max_mode ? feat_map_shape[3] / spatialScale : 1;

        size_t it = 0;
        for (const auto& input : cnnNetwork.getInputsInfo()) {
            const auto& info = input.second;
            InferenceEngine::Blob::Ptr blob;

            if (it == 1) {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                CommonTestUtils::fill_data_roi<InferenceEngine::Precision::FP32>(blob, feat_map_shape[0] - 1, height,
                                                                                 width, 1.0f, is_roi_max_mode);
            } else {
                blob = GenerateInput(*info);
            }
            inputs.push_back(blob);
            it++;
        }
    }
    void SkipBeforeLoad() override {
    }
    void SkipBeforeInfer() override {
        // [Track number: E#20262]
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::KmbSkipTestException("Bad results on Level0");
        }
    }
};

class KmbROIPoolingLayerTest_VPU3720 : public KmbROIPoolingLayerTest {};

TEST_P(KmbROIPoolingLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbROIPoolingLayerTest_VPU3720, SW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {{1, 3, 8, 8}, {3, 4, 50, 50}};

const std::vector<std::vector<size_t>> pooledShapes_max = {{1, 1}, {2, 2}, {3, 3}, {6, 6}};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {/*{1, 1},*/ {2, 2}, {3, 3}, {6, 6}};

const std::vector<std::vector<size_t>> coordShapes = {{1, 5}, /*{3, 5}, {5, 5}*/};

const std::vector<InferenceEngine::Precision> netPRCs = {InferenceEngine::Precision::FP16};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
        ::testing::ValuesIn(inShapes), ::testing::ValuesIn(coordShapes), ::testing::ValuesIn(pooledShapes_max),
        ::testing::ValuesIn(spatial_scales), ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
        ::testing::ValuesIn(netPRCs), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto test_ROIPooling_bilinear = ::testing::Combine(
        ::testing::ValuesIn(inShapes), ::testing::ValuesIn(coordShapes), ::testing::ValuesIn(pooledShapes_bilinear),
        ::testing::Values(spatial_scales[1]), ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),
        ::testing::ValuesIn(netPRCs), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_TestsROIPooling_max, KmbROIPoolingLayerTest, test_ROIPooling_max,
                         KmbROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, KmbROIPoolingLayerTest, test_ROIPooling_bilinear,
                         KmbROIPoolingLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, KmbROIPoolingLayerTest_VPU3720, test_ROIPooling_max,
                         KmbROIPoolingLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, KmbROIPoolingLayerTest_VPU3720, test_ROIPooling_bilinear,
                         KmbROIPoolingLayerTest::getTestCaseName);
