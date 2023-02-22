// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convolution.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXConvolutionLayerTest : public ConvolutionLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXConvolutionLayerTestLatency : public VPUXConvolutionLayerTest {};
class KmbConvolutionLayerTest : public VPUXConvolutionLayerTest {};

using VPUXConvolutionLayerTest_VPU3720_ELF = VPUXConvolutionLayerTest;

// Comparisons fail (ticket???)
TEST_P(KmbConvolutionLayerTest, DISABLED_CompareWithRefs) {
    Run();
}

TEST_P(KmbConvolutionLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbConvolutionLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXConvolutionLayerTest, CompareWithRefs_MLIR_VPU3720) {
    setPlatformVPU3720();
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXConvolutionLayerTest_VPU3720_ELF, CompareWithRefs_MLIR_VPU3720) {
    setPlatformVPU3720();
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    setSingleClusterMode();
    useELFCompilerBackend();
    Run();
}

TEST_P(VPUXConvolutionLayerTestLatency, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setPerformanceHintLatency();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace InferenceEngine;
using namespace LayerTestsDefinitions;

namespace {

/* ============= 1D Convolution ============= */

const auto conv1DParams = ::testing::Combine(::testing::ValuesIn<SizeVector>({{1}, {5}}),              // kernels
                                             ::testing::ValuesIn<SizeVector>({{1}, {3}}),              // strides
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}, {3}}),  // padBegins
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}, {2}}),  // padEnds
                                             ::testing::ValuesIn<SizeVector>({{1}, {2}}),              // dilations
                                             ::testing::Values(1, 4),                                  // numOutChannels
                                             ::testing::Values(ngraph::op::PadType::EXPLICIT)          // padType
);

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Convolution1D, KmbConvolutionLayerTest,
                        ::testing::Combine(conv1DParams,
                                           ::testing::Values(Precision::FP16),              // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),       // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),       // outPrc
                                           ::testing::Values(Layout::ANY),                  // inLayout
                                           ::testing::Values(Layout::ANY),                  // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AutoPadValid ============= */

const auto conv2DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{1, 1}, {3, 3}}),      // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(8, 16, 24, 32),                       // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Convolution2D_AutoPadValid, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AutoPadValid,                  //
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
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / CMajorCompatible ============= */

const auto conv2DParams_CMajorCompatible =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(8, 16),                               // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_CMajorCompatible, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_CMajorCompatible,                      //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / ExplicitPadding ============= */

const auto conv2DParams_ExplicitPadding =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),                                      // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),                              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {0, 1}, {0, 2}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {0, 1}}),          // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                      // dilations
                           ::testing::Values(1),                             // numOutChannels
                           ::testing::Values(ngraph::op::PadType::EXPLICIT)  // padType
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Convolution2D_ExplicitPadding, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_ExplicitPadding,                       //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AsymmetricPadding ============= */

const auto conv2DParams_AsymmetricPadding =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{5, 5}}),                                      // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {1, 2}, {2, 2}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {1, 2}, {2, 2}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                      // dilations
                           ::testing::Values(1),                             // numOutChannels
                           ::testing::Values(ngraph::op::PadType::EXPLICIT)  // padType
        );

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_Convolution2D_AsymmetricPadding, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AsymmetricPadding,                     //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AsymmetricKernel ============= */

const auto conv2DParams_AsymmetricKernel =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 1}, {1, 3}}),      // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AsymmetricKernel, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AsymmetricKernel,                      //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AsymmetricStrides ============= */

const auto conv2DParams_AsymmetricStrides =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 2}, {2, 1}}),      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::EXPLICIT)        // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AsymmetricStrides, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AsymmetricStrides,                     //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AsymmetricStrides, VPUXConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AsymmetricStrides,                     //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / Dilated ============= */

const auto conv2DParams_Dilated =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),              // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_Dilated, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_Dilated,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeSize ============= */

const auto conv2DParams_LargeSize1 =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(64),                                  // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_LargeSize1, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_LargeSize1,                               //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 128, 128}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_LargeSize2 =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(16),                                  // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

const auto conv2DParams_LargeSize2_ELF =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{1, 1}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(16),                                  // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_LargeSize2, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_LargeSize2,                               //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 256, 256}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeStride ============= */

const auto conv2DParams_LargeStrides =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{11, 11}, {2, 2}}),    // kernels
                           ::testing::ValuesIn<SizeVector>({{11, 11}, {10, 10}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(16),                                  // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_LargeStrides, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_LargeStrides,                          //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::FP16),                 // inPrc
                                           ::testing::Values(Precision::FP16),                 // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / SOK ============= */

const auto conv2DParams_SOK = ::testing::Combine(::testing::ValuesIn<SizeVector>({{1, 1}}),              // kernels
                                                 ::testing::ValuesIn<SizeVector>({{1, 1}}),              // strides
                                                 ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                                                 ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                                                 ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                                                 ::testing::Values(64),                         // numOutChannels
                                                 ::testing::Values(ngraph::op::PadType::VALID)  // padType
);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_SOK, VPUXConvolutionLayerTestLatency,
                        ::testing::Combine(conv2DParams_SOK,                                  //
                                           ::testing::Values(Precision::FP16),                // netPrc
                                           ::testing::Values(Precision::FP16),                // inPrc
                                           ::testing::Values(Precision::FP16),                // outPrc
                                           ::testing::Values(Layout::ANY),                    // inLayout
                                           ::testing::Values(Layout::ANY),                    // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 32, 3, 3}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

/* ============= ELF ============= */

INSTANTIATE_TEST_CASE_P(smoke_precommit_Convolution2D, VPUXConvolutionLayerTest_VPU3720_ELF,
                        ::testing::Combine(conv2DParams_LargeSize2,
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::NCHW),                     // inLayout
                                           ::testing::Values(Layout::NCHW),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution2D, VPUXConvolutionLayerTest_VPU3720_ELF,
                        ::testing::Combine(conv2DParams_LargeSize2,
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::NCHW, Layout::NHWC),       // inLayout
                                           ::testing::Values(Layout::NCHW, Layout::NHWC),       // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

}  // namespace
