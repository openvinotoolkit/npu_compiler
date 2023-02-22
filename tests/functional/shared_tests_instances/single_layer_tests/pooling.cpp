// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/pooling.hpp"
#include "vpux_private_config.hpp"

#include <vector>

#include <common/functions.h>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPoolingLayerTest : public PoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto& poolParams = std::get<0>(GetParam());

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> strides;
        ngraph::op::RoundingType roundingMode;
        std::tie(poolType, std::ignore, strides, std::ignore, std::ignore, roundingMode, std::ignore, std::ignore) =
                poolParams;

        if (poolType == ngraph::helpers::PoolingTypes::AVG && isCompilerMLIR() &&
            configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] == "DefaultHW") {
            threshold = 0.25;
        }

        // MLIR uses software layer, which seem to be flawed
        if (poolType == ngraph::helpers::PoolingTypes::AVG) {
            if (strides[0] != 1 || strides[1] != 1) {
                throw LayerTestsUtils::KmbSkipTestException("AVG pool strides != 1 produces inaccurate results");
            }
        }
    }

    void SkipBeforeInfer() override {
    }
};

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

class VPUXPoolingLayerTest_VPU3720 : public PoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(VPUXPoolingLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXPoolingLayerTest_VPU3720, CompareWithRefs_MLIR_HW_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

using VPUXPoolingLayerTest_VPU3720_SingleCluster = VPUXPoolingLayerTest_VPU3720;

TEST_P(VPUXPoolingLayerTest_VPU3720_SingleCluster, CompareWithRefs_MLIR_HW_VPU3720) {
    setPlatformVPU3720();
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    setSingleClusterMode();
    useELFCompilerBackend();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace InferenceEngine;
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

/* ============= AutoPadValid ============= */

const auto pool_AutoPadValid = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                  ::testing::ValuesIn<SizeVector>({{3, 3}, {5, 5}}),        // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),        // strides
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),       //
                                                  ::testing::Values(ngraph::op::PadType::VALID),            //
                                                  ::testing::Values(false)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Pooling_AutoPadValid, KmbPoolingLayerTest,
                         ::testing::Combine(pool_AutoPadValid,                          //
                                            ::testing::Values(Precision::FP16),         // netPrc
                                            ::testing::Values(Precision::UNSPECIFIED),  // inPrc
                                            ::testing::Values(Precision::UNSPECIFIED),  // outPrc
                                            ::testing::Values(Layout::ANY),             // inLayout
                                            ::testing::Values(Layout::ANY),             // outLayout
                                            ::testing::ValuesIn<SizeVector>({{1, 8, 32, 32},
                                                                             {1, 16, 24, 24},
                                                                             {1, 24, 16, 16},
                                                                             {1, 32, 8, 8}}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                         PoolingLayerTest::getTestCaseName);

/* ============= ExplicitPadding ============= */

const auto pool_ExplicitPadding =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),    //
                           ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // kernels
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),                  // strides
                           ::testing::ValuesIn<SizeVector>({{0, 0}, {1, 1}, {0, 1}}),  // padBegins
                           ::testing::ValuesIn<SizeVector>({{0, 0}, {1, 1}, {0, 1}}),  // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR, ngraph::op::RoundingType::CEIL),  //
                           ::testing::Values(ngraph::op::PadType::EXPLICIT),                                    //
                           ::testing::Values(false)  // excludePad
        );

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Pooling_ExplicitPadding, KmbPoolingLayerTest,
                         ::testing::Combine(pool_ExplicitPadding,                                //
                                            ::testing::Values(Precision::FP16),                  // netPrc
                                            ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                            ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                            ::testing::Values(Layout::ANY),                      // inLayout
                                            ::testing::Values(Layout::ANY),                      // outLayout
                                            ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                         PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricKernel ============= */

const auto pool_AsymmetricKernel = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                      ::testing::ValuesIn<SizeVector>({{3, 1}, {1, 3}}),   // kernels
                                                      ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),   // strides
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                      ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                      ::testing::Values(ngraph::op::PadType::VALID),       //
                                                      ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Pooling_AsymmetricKernel, KmbPoolingLayerTest,
                         ::testing::Combine(pool_AsymmetricKernel,                               //
                                            ::testing::Values(Precision::FP16),                  // netPrc
                                            ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                            ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                            ::testing::Values(Layout::ANY),                      // inLayout
                                            ::testing::Values(Layout::ANY),                      // outLayout
                                            ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                         PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricStrides ============= */

const auto pool_AsymmetricStrides = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                       ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                       ::testing::ValuesIn<SizeVector>({{1, 2}, {2, 1}}),   // strides
                                                       ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                       ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                       ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                       ::testing::Values(ngraph::op::PadType::VALID),       //
                                                       ::testing::Values(false)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Pooling_AsymmetricStrides, KmbPoolingLayerTest,
                         ::testing::Combine(pool_AsymmetricStrides,                              //
                                            ::testing::Values(Precision::FP16),                  // netPrc
                                            ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                            ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                            ::testing::Values(Layout::ANY),                      // inLayout
                                            ::testing::Values(Layout::ANY),                      // outLayout
                                            ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         PoolingLayerTest::getTestCaseName);

/* ============= LargeSize ============= */

const auto pool_LargeSize1 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                ::testing::ValuesIn<SizeVector>({{2, 2}}),           // strides
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                ::testing::Values(ngraph::op::PadType::VALID),       //
                                                ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargeSize1, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargeSize1,                                       //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 64, 128, 128}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargeSize2 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                ::testing::ValuesIn<SizeVector>({{2, 2}}),           // strides
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                ::testing::Values(ngraph::op::PadType::VALID),       //
                                                ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargeSize2, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargeSize2,                                       //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 256, 256}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= LargeStrides ============= */

const auto pool_LargeStrides = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{3, 3}, {11, 11}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{9, 9}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargeStrides, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargeStrides,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= Padding valitation ( > K_SZ/2) ============= */

const auto pool_LargePadding2 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                   ::testing::ValuesIn<SizeVector>({{2, 2}, {3, 3}}),   // kernels
                                                   ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                   ::testing::ValuesIn<SizeVector>({{2, 2}}),           // padBegins
                                                   ::testing::ValuesIn<SizeVector>({{2, 2}}),           // padEnds
                                                   ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                   ::testing::Values(ngraph::op::PadType::VALID),       //
                                                   ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding2, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding2,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding3 =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                       //
                           ::testing::ValuesIn<SizeVector>({{3, 3}, {4, 4}, {5, 5}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                  // strides
                           ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // padBegins
                           ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),         //
                           ::testing::Values(ngraph::op::PadType::VALID),              //
                           ::testing::Values(false)                                    // excludePad
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding3, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding3,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding4 =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                               //
                           ::testing::ValuesIn<SizeVector>({{4, 4}, {5, 5}, {6, 6}, {7, 7}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                          // strides
                           ::testing::ValuesIn<SizeVector>({{4, 4}}),                          // padBegins
                           ::testing::ValuesIn<SizeVector>({{4, 4}}),                          // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                 //
                           ::testing::Values(ngraph::op::PadType::VALID),                      //
                           ::testing::Values(false)                                            // excludePad
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding4, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding4,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding5 =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                       //
                           ::testing::ValuesIn<SizeVector>({{5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                  // strides
                           ::testing::ValuesIn<SizeVector>({{5, 5}}),                                  // padBegins
                           ::testing::ValuesIn<SizeVector>({{5, 5}}),                                  // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                         //
                           ::testing::Values(ngraph::op::PadType::VALID),                              //
                           ::testing::Values(false)                                                    // excludePad
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding5, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding5,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding6 = ::testing::Combine(
        ::testing::Values(PoolingTypes::MAX),                                                   //
        ::testing::ValuesIn<SizeVector>({{6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
        ::testing::ValuesIn<SizeVector>({{1, 1}}),                                              // strides
        ::testing::ValuesIn<SizeVector>({{6, 6}}),                                              // padBegins
        ::testing::ValuesIn<SizeVector>({{6, 6}}),                                              // padEnds
        ::testing::Values(ngraph::op::RoundingType::FLOOR),                                     //
        ::testing::Values(ngraph::op::PadType::VALID),                                          //
        ::testing::Values(false)                                                                // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding6, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding6,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding7 =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                           //
                           ::testing::ValuesIn<SizeVector>({{7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                      // strides
                           ::testing::ValuesIn<SizeVector>({{7, 7}}),                                      // padBegins
                           ::testing::ValuesIn<SizeVector>({{7, 7}}),                                      // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                             //
                           ::testing::Values(ngraph::op::PadType::VALID),                                  //
                           ::testing::Values(false)                                                        // excludePad
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding7, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding7,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding8 =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                   //
                           ::testing::ValuesIn<SizeVector>({{8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                              // strides
                           ::testing::ValuesIn<SizeVector>({{8, 8}}),                              // padBegins
                           ::testing::ValuesIn<SizeVector>({{8, 8}}),                              // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                     //
                           ::testing::Values(ngraph::op::PadType::VALID),                          //
                           ::testing::Values(false)                                                // excludePad
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargePadding8, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding8,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */

const auto avgPool_largeKernels = ::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                                     ::testing::ValuesIn<SizeVector>({{23, 30}}),         // kernels
                                                     ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                     ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                     ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                     ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                     ::testing::Values(ngraph::op::PadType::VALID),       //
                                                     ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_AvgPooling_LargeKernels, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_largeKernels,                                  //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 2048, 23, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsX ============= */

const auto avgPool_largeKernelsX = ::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                                      ::testing::ValuesIn<SizeVector>({{1, 14}}),          // kernels
                                                      ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                      ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                      ::testing::Values(ngraph::op::PadType::VALID),       //
                                                      ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_AvgPooling_LargeKernelsX, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_largeKernelsX,                              //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 1, 14}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsY ============= */

const auto avgPool_largeKernelsY = ::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                                      ::testing::ValuesIn<SizeVector>({{14, 1}}),          // kernels
                                                      ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                      ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                      ::testing::Values(ngraph::op::PadType::VALID),       //
                                                      ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_AvgPooling_LargeKernelsY, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_largeKernelsY,                              //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 14, 1}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large Kernels ============= */

const auto maxPool_largeKernels = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                     ::testing::ValuesIn<SizeVector>({{23, 30}}),         // kernels
                                                     ::testing::ValuesIn<SizeVector>({{23, 30}}),         // strides
                                                     ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                     ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                     ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                     ::testing::Values(ngraph::op::PadType::VALID),       //
                                                     ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_MaxPooling_LargeKernels, KmbPoolingLayerTest,
                        ::testing::Combine(maxPool_largeKernels,                                  //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 2048, 23, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsX ============= */

const auto maxPool_largeKernelsX = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                      ::testing::ValuesIn<SizeVector>({{1, 14}}),          // kernels
                                                      ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                      ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                      ::testing::Values(ngraph::op::PadType::VALID),       //
                                                      ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_MaxPooling_LargeKernelsX, KmbPoolingLayerTest,
                        ::testing::Combine(maxPool_largeKernelsX,                              //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 1, 14}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsY ============= */

const auto maxPool_largeKernelsY = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                      ::testing::ValuesIn<SizeVector>({{14, 1}}),          // kernels
                                                      ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                      ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                      ::testing::Values(ngraph::op::PadType::VALID),       //
                                                      ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_MaxPooling_LargeKernelsY, KmbPoolingLayerTest,
                        ::testing::Combine(maxPool_largeKernelsY,                              //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 14, 1}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= AvgPooling / Exclude_Pad Handling ============= */

const auto avgPool_excludePad = ::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                                   ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                   ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                                   ::testing::ValuesIn<SizeVector>({{1, 1}}),           // padBegins
                                                   ::testing::ValuesIn<SizeVector>({{1, 1}}),           // padEnds
                                                   ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                   ::testing::Values(ngraph::op::PadType::VALID),       //
                                                   ::testing::Values(true)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_avgPool_excludePad, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_excludePad,                                  //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 28, 28}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= VPU 3720 ============= */

const auto pool_ExplicitNoPadding = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                                                       ::testing::ValuesIn<SizeVector>({{14, 14}}),  // kernels
                                                       ::testing::Values<SizeVector>({1, 1}),        // strides
                                                       ::testing::Values<SizeVector>({0, 0}),        // padBegins
                                                       ::testing::Values<SizeVector>({0, 0}),        // padEnds
                                                       ::testing::Values(ngraph::op::RoundingType::FLOOR),
                                                       ::testing::Values(ngraph::op::PadType::EXPLICIT),
                                                       ::testing::Values(true)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding_VPU3720, VPUXPoolingLayerTest_VPU3720,
                         ::testing::Combine(pool_ExplicitNoPadding,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NCHW),                 // inLayout
                                            ::testing::Values(Layout::NCHW),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 30, 14, 14}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding_VPU3720_ELF, VPUXPoolingLayerTest_VPU3720_SingleCluster,
                         ::testing::Combine(pool_ExplicitNoPadding,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NCHW),                 // inLayout
                                            ::testing::Values(Layout::NCHW),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 30, 14, 14}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_NHWC_NoPadding_VPU3720, VPUXPoolingLayerTest_VPU3720,
                         ::testing::Combine(pool_ExplicitNoPadding,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NHWC),                 // inLayout
                                            ::testing::Values(Layout::NHWC),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 30, 14, 14}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NHWC_NoPadding_VPU3720_ELF, VPUXPoolingLayerTest_VPU3720_SingleCluster,
                         ::testing::Combine(pool_ExplicitNoPadding,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NHWC),                 // inLayout
                                            ::testing::Values(Layout::NHWC),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 30, 14, 14}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

// [Track number: E#58512]
const auto pool_ExplicitNoPadding_failing_case = ::testing::Combine(
        ::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn<SizeVector>({{14, 1}, {1, 14}}),  // kernels
        ::testing::Values<SizeVector>({1, 1}),                                                      // strides
        ::testing::Values<SizeVector>({0, 0}),                                                      // padBegins
        ::testing::Values<SizeVector>({0, 0}),                                                      // padEnds
        ::testing::Values(ngraph::op::RoundingType::FLOOR), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Pooling_Fail_NHWC_NoPadding_VPU3720, VPUXPoolingLayerTest_VPU3720,
                         ::testing::Combine(pool_ExplicitNoPadding_failing_case,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NHWC),                 // inLayout
                                            ::testing::Values(Layout::NHWC),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 30, 14, 14}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

// U-net usecase
const auto pool_unet = ::testing::Combine(
        ::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG), ::testing::Values<SizeVector>({12, 1}),  // kernels
        ::testing::Values<SizeVector>({1, 1}),                                                            // strides
        ::testing::Values<SizeVector>({0, 0}),                                                            // padBegins
        ::testing::Values<SizeVector>({0, 0}),                                                            // padEnds
        ::testing::Values(ngraph::op::RoundingType::FLOOR), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(true)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_unet_VPU3720, VPUXPoolingLayerTest_VPU3720,
                         ::testing::Combine(pool_unet,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NCHW),                 // inLayout
                                            ::testing::Values(Layout::NCHW),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 1, 12, 176}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_unet_VPU3720_ELF, VPUXPoolingLayerTest_VPU3720_SingleCluster,
                         ::testing::Combine(pool_unet,
                                            ::testing::Values(Precision::FP16),              // netPrc
                                            ::testing::Values(Precision::FP16),              // inPrc
                                            ::testing::Values(Precision::FP16),              // outPrc
                                            ::testing::Values(Layout::NCHW),                 // inLayout
                                            ::testing::Values(Layout::NCHW),                 // outLayout
                                            ::testing::Values<SizeVector>({1, 1, 12, 176}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXPoolingLayerTest_VPU3720::getTestCaseName);

// large kernel
const auto pooling_largeKernel_VPU3720 = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                            ::testing::ValuesIn<SizeVector>({{28, 28}}),  // kernels
                                                            ::testing::ValuesIn<SizeVector>({{1, 1}}),    // strides
                                                            ::testing::ValuesIn<SizeVector>({{0, 0}}),    // padBegins
                                                            ::testing::ValuesIn<SizeVector>({{0, 0}}),    // padEnds
                                                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                            ::testing::Values(ngraph::op::PadType::VALID),       //
                                                            ::testing::Values(true)  // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargeKernel3720, VPUXPoolingLayerTest_VPU3720,
                        ::testing::Combine(pooling_largeKernel_VPU3720,                         //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 70, 28, 28}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXPoolingLayerTest_VPU3720::getTestCaseName);

// These tests are temporarily disabled because they are failing on older MV Tools versions
// They will be enabled back when E#64778 gets completed.
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Pooling_LargeKernel3720_ELF, VPUXPoolingLayerTest_VPU3720_SingleCluster,
                        ::testing::Combine(pooling_largeKernel_VPU3720,                         //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 70, 28, 28}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXPoolingLayerTest_VPU3720::getTestCaseName);

// AutoPadValid
const auto pool_AutoPadValid_VPU3720 = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                          ::testing::ValuesIn<SizeVector>({{3, 3}, {5, 5}}),  // kernels
                                                          ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),  // strides
                                                          ::testing::ValuesIn<SizeVector>({{0, 0}}),  // padBegins
                                                          ::testing::ValuesIn<SizeVector>({{0, 0}}),  // padEnds
                                                          ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                          ::testing::Values(ngraph::op::PadType::VALID),       //
                                                          ::testing::Values(false)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AutoPadValid_VPU3720_ELF, VPUXPoolingLayerTest_VPU3720_SingleCluster,
                         ::testing::Combine(pool_AutoPadValid_VPU3720,           //
                                            ::testing::Values(Precision::FP16),  // netPrc
                                            ::testing::Values(Precision::FP16),  // inPrc
                                            ::testing::Values(Precision::FP16),  // outPrc
                                            ::testing::Values(Layout::NHWC),     // inLayout
                                            ::testing::Values(Layout::NHWC),     // outLayout
                                            ::testing::ValuesIn<SizeVector>({{1, 8, 32, 32},
                                                                             {1, 16, 24, 24},
                                                                             {1, 24, 16, 16},
                                                                             {1, 32, 8, 8}}),  // inputShapes
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                         PoolingLayerTest::getTestCaseName);

}  // namespace
