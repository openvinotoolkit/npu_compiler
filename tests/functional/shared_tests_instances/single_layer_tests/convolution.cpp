// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/convolution.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConvolutionLayerTestCommon : public ConvolutionLayerTest, virtual public VpuOv2LayerTest {};

class ConvolutionLayerTest_NPU3720_SCM : public ConvolutionLayerTestCommon {};
class ConvolutionLayerTest_NPU3720_HW : public ConvolutionLayerTestCommon {};
class ConvolutionLayerTest_NPU3720_SW : public ConvolutionLayerTestCommon {};
class ConvolutionLayerTestLatency_NPU3720 : public ConvolutionLayerTestCommon {};

class ConvolutionLayerTest_NPU4000_SW : public ConvolutionLayerTestCommon {};
class ConvolutionLayerTest_NPU4000_HW : public ConvolutionLayerTestCommon {};

class ConvolutionLayerTest_FP32_SW : public ConvolutionLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

class ConvolutionLayerTest_NPU3720_MULTI_BATCH_HW : public ConvolutionLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "skip-unroll-batch=true";
    }
};

class ConvolutionLayerTest_MULTI_BATCH_HW : public ConvolutionLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::batch_compiler_mode_settings.name()] =
                "batch-compile-method=unroll batch-unroll-settings={skip-unroll-batch=true}";
    }
};

// NPU3720
TEST_P(ConvolutionLayerTest_NPU3720_HW, HW) {
    rel_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionLayerTest_NPU3720_SW, SW) {
    rel_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionLayerTest_FP32_SW, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionLayerTest_NPU3720_SCM, HW) {
    setDefaultHardwareMode();
    setSingleClusterMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionLayerTestLatency_NPU3720, HW) {
    rel_threshold = 0.02;
    setPerformanceHintLatency();
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionLayerTest_MULTI_BATCH_HW, NPU3720_HW) {
    rel_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

// NPU4000
TEST_P(ConvolutionLayerTest_NPU4000_SW, SW) {
    rel_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvolutionLayerTest_FP32_SW, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvolutionLayerTest_NPU4000_HW, HW) {
    rel_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvolutionLayerTest_MULTI_BATCH_HW, NPU4000_HW) {
    rel_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

/* ============= 1D Convolution ============= */

const auto conv1DParams = ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1}, {5}}),     // kernels
                                             ::testing::ValuesIn<std::vector<size_t>>({{1}, {3}}),     // strides
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}, {3}}),  // padBegins
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}, {2}}),  // padEnds
                                             ::testing::ValuesIn<std::vector<size_t>>({{1}, {2}}),     // dilations
                                             ::testing::Values(1, 4),                                  // numOutChannels
                                             ::testing::Values(ov::op::PadType::EXPLICIT)              // padType
);

const auto conv1D = ::testing::Combine(
        conv1DParams,
        ::testing::Values(ov::element::f16),                                                  // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 16, 64}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution1D, ConvolutionLayerTest_NPU3720_SW, conv1D,
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution1D, ConvolutionLayerTest_NPU4000_SW, conv1D,
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 1D Convolution / LargeKernel ============= */
const auto conv1DParams_LargeKernel =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{512}}),   // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{160}}),   // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1}}),     // dilations
                           ::testing::Values(257),                              // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)         // padType
        );

const auto convLargeKernel1D = ::testing::Combine(
        conv1DParams_LargeKernel,
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 1, 80000}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution1D_LargeKernel, ConvolutionLayerTest_NPU3720_HW, convLargeKernel1D,
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution1D_LargeKernel, ConvolutionLayerTest_NPU4000_HW, convLargeKernel1D,
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AutoPadValid ============= */

const auto conv2DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),       // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),       // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),          // dilations
                           ::testing::Values(8, 16, 24, 32),                            // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)                    // padType
        );
std::vector<std::vector<ov::Shape>> iShape2D = {
        {{1, 8, 32, 32}}, {{1, 16, 24, 24}}, {{1, 24, 16, 16}}, {{1, 32, 8, 8}}};

const auto conv2D_AutoPadValid = ::testing::Combine(conv2DParams_AutoPadValid,            //
                                                    ::testing::Values(ov::element::f16),  // netPrc
                                                    ::testing::ValuesIn(static_shapes_to_test_representation(iShape2D)),
                                                    ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest_NPU3720_SW, conv2D_AutoPadValid,
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest_NPU4000_SW, conv2D_AutoPadValid,
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 2D Convolution / CMajorCompatible ============= */

const auto conv2DParams_CMajorCompatible =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(8, 16),                               // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)               // padType
        );

const auto conv2D_CMajorCompatible = ::testing::Combine(
        conv2DParams_CMajorCompatible,
        ::testing::Values(ov::element::f16),                                                     // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 64, 64}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_CMajorCompatible, ConvolutionLayerTest_NPU3720_SW, conv2D_CMajorCompatible,
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_CMajorCompatible, ConvolutionLayerTest_NPU4000_SW, conv2D_CMajorCompatible,
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 3D Convolution / 3x2x2 Kernel ============= */

const auto conv3DParams_3x2x2_Kernel =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 2, 2}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),     // dilations
                           ::testing::Values(32),                                     // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)                  // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_3x2x2_Kernel, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv3DParams_3x2x2_Kernel,            //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 32, 5, 28, 28}})}),                  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

// NPU4000
INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_3x2x2_Kernel, ConvolutionLayerTest_NPU4000_HW,
                         ::testing::Combine(conv3DParams_3x2x2_Kernel,            //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 32, 5, 28, 28}})}),                  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution / 3x1x1 Kernel ============= */

const auto conv3DParams_3x1x1_Kernel =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 1, 1}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),     // dilations
                           ::testing::Values(32),                                     // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)                  // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_3x1x1_Kernel, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv3DParams_3x1x1_Kernel,            //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 32, 6, 28, 28}})}),                  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

// NPU4000
INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_3x1x1_Kernel, ConvolutionLayerTest_NPU4000_HW,
                         ::testing::Combine(conv3DParams_3x1x1_Kernel,            //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 32, 6, 28, 28}})}),                  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeKernel ============= */

const auto conv2DParams_LargeKernel1 =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{13, 13}}),   // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(8),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)               // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeKernel, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_LargeKernel1,            //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 64, 64}})}),                      // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeDilations ============= */

const auto conv2DParams_LargeDilations =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{7, 7}}),     // dilations
                           ::testing::Values(8),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)               // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeDilations, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_LargeDilations,          //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 64, 64}})}),                      // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 2D Convolution / ExplicitPadding ============= */

const auto conv2DParams_ExplicitPadding =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                             // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),                     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {0, 1}, {0, 2}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {0, 1}}),          // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                             // dilations
                           ::testing::Values(1),                         // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)  // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ExplicitPadding, ConvolutionLayerTest_NPU3720_SW,
                         ::testing::Combine(conv2DParams_ExplicitPadding,         //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 16, 16}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AsymmetricPadding ============= */

const auto conv2DParams_AsymmetricPadding =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{5, 5}}),                             // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                             // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {1, 2}, {2, 2}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {1, 2}, {2, 2}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                             // dilations
                           ::testing::Values(1),                         // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)  // padType
        );

const auto conv2D_AsymmetricPadding = ::testing::Combine(
        conv2DParams_AsymmetricPadding,                                                          //
        ::testing::Values(ov::element::f16),                                                     // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 64, 64}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AsymmetricPadding, ConvolutionLayerTest_NPU3720_SW,
                         conv2D_AsymmetricPadding, ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Convolution2D_AsymmetricPadding, ConvolutionLayerTest_NPU4000_SW,
                         conv2D_AsymmetricPadding, ConvolutionLayerTest::getTestCaseName);
/* ============= 2D Convolution / AsymmetricKernel ============= */

const auto conv2DParams_AsymmetricKernel =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 1}, {1, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),       // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),       // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),          // dilations
                           ::testing::Values(1),                                        // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)                    // padType
        );

const auto conv2D_AsymmetricKernel = ::testing::Combine(
        conv2DParams_AsymmetricKernel,                                                           //
        ::testing::Values(ov::element::f16),                                                     // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 16, 16}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AsymmetricKernel, ConvolutionLayerTest_NPU3720_SW, conv2D_AsymmetricKernel,
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AsymmetricKernel, ConvolutionLayerTest_NPU4000_SW, conv2D_AsymmetricKernel,
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 2D Convolution / AsymmetricStrides ============= */

const auto conv2DParams_AsymmetricStrides =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),          // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 2}, {2, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),       // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),       // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),          // dilations
                           ::testing::Values(1),                                        // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)                 // padType
        );

const auto conv2D_AsymmetricStrides = ::testing::Combine(
        conv2DParams_AsymmetricStrides,                                                          //
        ::testing::Values(ov::element::f16),                                                     // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 16, 16}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AsymmetricStrides, ConvolutionLayerTest_NPU3720_HW,
                         conv2D_AsymmetricStrides, ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_AsymmetricStrides2 =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 4}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 4}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(64),                                  // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)            // padType
        );

std::vector<std::vector<ov::Shape>> iShape = {{{1, 1, 1, 256}}, {{64, 1, 1, 4}}};
const auto conv2D_AsymmetricStrides2 =
        ::testing::Combine(conv2DParams_AsymmetricStrides2,                                    //
                           ::testing::Values(ov::element::f16),                                // netPrc
                           ::testing::ValuesIn(static_shapes_to_test_representation(iShape)),  // inputShapes
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AsymmetricStrides2, ConvolutionLayerTest_NPU3720_HW,
                         conv2D_AsymmetricStrides2, ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeKernel ============= */

const auto conv2DParams_LargeKernel =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{22, 22}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{16, 16}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{16, 16}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{16, 16}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),       // dilations
                           ::testing::Values(1),                                     // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)              // padType
        );
std::vector<std::vector<ov::Shape>> iShapeLargeKernel = {{{1, 1, 320, 320}}, {{1, 3, 320, 320}}};
const auto conv2D_LargeKernel =
        ::testing::Combine(conv2DParams_LargeKernel,                                                      //
                           ::testing::Values(ov::element::f16),                                           // netPrc
                           ::testing::ValuesIn(static_shapes_to_test_representation(iShapeLargeKernel)),  // inputShapes
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeKernel_Explicit, ConvolutionLayerTest_NPU3720_HW, conv2D_LargeKernel,
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeKernel / OneDim ============= */
const auto conv2DParams_LargeKernel_OneDim =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 512}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{128, 128}}),   // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 128}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 128}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),       // dilations
                           ::testing::Values(258),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)              // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeKernel_OneDim, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_LargeKernel_OneDim,      //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 1, 1, 2176}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / Dilated ============= */

const auto conv2DParams_Dilated =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),     // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)               // padType
        );

const auto conv2D_Dilated = ::testing::Combine(
        conv2DParams_Dilated,                                                                    //
        ::testing::Values(ov::element::f16),                                                     // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 16, 16}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_Dilated, ConvolutionLayerTest_NPU3720_SW, conv2D_Dilated,
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_Dilated, ConvolutionLayerTest_NPU4000_SW, conv2D_Dilated,
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 2D Convolution / LargeSize ============= */

const auto conv2DParams_LargeSize1 =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(64),                                  // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)               // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeSize1, ConvolutionLayerTest_NPU3720_SW,
                         ::testing::Combine(conv2DParams_LargeSize1,              //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 16, 128, 128}})}),                   // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_LargeSize2 =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(16),                                  // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)               // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeSize2, ConvolutionLayerTest_NPU3720_SW,
                         ::testing::Combine(conv2DParams_LargeSize2,              //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 16, 256, 256}})}),                   // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_LargeSize3 = ::testing::Combine(::testing::Values<std::vector<size_t>>({1, 1}),     // kernels
                                                        ::testing::Values<std::vector<size_t>>({1, 1}),     // strides
                                                        ::testing::Values<std::vector<ptrdiff_t>>({0, 0}),  // padsBegin
                                                        ::testing::Values<std::vector<ptrdiff_t>>({0, 0}),  // padsEnd
                                                        ::testing::Values<std::vector<size_t>>({1, 1}),     // dilations
                                                        ::testing::Values(96),  // numOutChannels
                                                        ::testing::Values(ov::op::PadType::EXPLICIT));  // padType

const auto conv2DInstantiateParams_LargeSize3 = ::testing::Combine(
        conv2DParams_LargeSize3,
        ::testing::Values(ov::element::f32),                                                      // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 9216, 1, 1}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeSize3, ConvolutionLayerTest_FP32_SW,
                         conv2DInstantiateParams_LargeSize3, ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeStride ============= */

const auto conv2DParams_LargeStrides =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{11, 11}, {2, 2}}),    // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{11, 11}, {10, 10}}),  // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),           // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),           // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),              // dilations
                           ::testing::Values(16),                                           // numOutChannels
                           ::testing::Values(ov::op::PadType::VALID)                        // padType
        );

const auto conv2D_LargeStrides = ::testing::Combine(
        conv2DParams_LargeStrides,
        ::testing::Values(ov::element::f16),                                                     // netPrc
        ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 3, 64, 64}})}),  // inputShapes
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_LargeStrides, ConvolutionLayerTest_NPU3720_SW, conv2D_LargeStrides,
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / SOK ============= */

const auto conv2DParams_SOK = ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // kernels
                                                 ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                                                 ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                                                 ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                                                 ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                                                 ::testing::Values(64),                     // numOutChannels
                                                 ::testing::Values(ov::op::PadType::VALID)  // padType
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_SOK, ConvolutionLayerTestLatency_NPU3720,
                         ::testing::Combine(conv2DParams_SOK,                     //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 32, 3, 3}})}),                       // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= BatchN to Batch1 ============= */

const auto conv2DParams_NBatch = ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // kernels
                                                    ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                                                    ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                                                    ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                                                    ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                                                    ::testing::Values(64),                     // numOutChannels
                                                    ::testing::Values(ov::op::PadType::VALID)  // padType
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_NBatch, ConvolutionLayerTestLatency_NPU3720,
                         ::testing::Combine(conv2DParams_NBatch,                  //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    32, 32, 1, 3}})}),                      // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / SOB ============= */

const auto conv2DParams_SOB = ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                                                 ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                                                 ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padBegins
                                                 ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padEnds
                                                 ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                                                 ::testing::Values(1),                      // numOutChannels
                                                 ::testing::Values(ov::op::PadType::VALID)  // padType
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_SOB, ConvolutionLayerTest_MULTI_BATCH_HW,
                         ::testing::Combine(conv2DParams_SOB,                     //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    2, 3, 96, 96}})}),                      // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),  //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= SCM ============= */

// Disabled due to E-123991
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_Convolution2D, ConvolutionLayerTest_NPU3720_SCM,
                         ::testing::Combine(conv2DParams_LargeSize2,
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 16, 16, 16}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / ShapeCast ============= */
const auto conv2DParams_ShapeCast_PadBeginEnd =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(3),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)            // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ShapeCast_PadBeginEnd, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_ShapeCast_PadBeginEnd,   //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 1080, 2048}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_ShapeCast_PadBegin =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(3),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)            // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ShapeCast_PadBegin, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_ShapeCast_PadBegin,      //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 1080, 2048}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_ShapeCast_PadEnd =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(3),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)            // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ShapeCast_PadEnd, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_ShapeCast_PadEnd,        //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 1080, 2048}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);
const auto conv2DParams_ShapeCast_PadBeginEnd_Stride =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{4, 4}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{1, 1}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),     // dilations
                           ::testing::Values(3),                                   // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)            // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ShapeCast_PadBeginEnd_Stride, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv2DParams_ShapeCast_PadBeginEnd_Stride,  //
                                            ::testing::Values(ov::element::f16),        // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 1080, 2048}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

const auto conv3DParams =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),                // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),                // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0, 1}, {1, 0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0, 0}}),             // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),                // dilations
                           ::testing::Values(16),                                                // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)                          // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv3DParams,                         //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 64, 64, 64}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D, ConvolutionLayerTest_NPU4000_HW,
                         ::testing::Combine(conv3DParams,                         //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 64, 64, 64}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

// Conv 3D with strides not equal to 1
const auto conv3DComplexStrides =
        ::testing::Combine(::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 2, 2}}),     // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0, 0}}),  // padEnds
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1, 1}}),     // dilations
                           ::testing::Values(16),                                     // numOutChannels
                           ::testing::Values(ov::op::PadType::EXPLICIT)               // padType
        );

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_ComplexStrides, ConvolutionLayerTest_NPU3720_HW,
                         ::testing::Combine(conv3DComplexStrides,                 //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 64, 64, 64}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_ComplexStrides, ConvolutionLayerTest_NPU4000_HW,
                         ::testing::Combine(conv3DComplexStrides,                 //
                                            ::testing::Values(ov::element::f16),  // netPrc
                                            ::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{
                                                    1, 3, 64, 64, 64}})}),  // inputShapes
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         ConvolutionLayerTest::getTestCaseName);
}  // namespace
