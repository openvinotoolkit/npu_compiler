//
// Copyright (C) 2019-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/interpolate.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using ov::op::util::InterpolateBase;

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(InterpolateLayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Interpolate11LayerTest);

class InterpolateLayerTestCommon : public InterpolateLayerTest, virtual public VpuOv2LayerTest {};

// Per-platform fixture subclasses used for suites with limited platform coverage.
// Full-coverage suites use InterpolateLayerTestCommon directly.
class InterpolateLayerTest_NPU3720 : public InterpolateLayerTestCommon {};
class InterpolateLayerTest_NPU4000 : public InterpolateLayerTestCommon {};
class InterpolateLayerTest_NPU5010 : public InterpolateLayerTestCommon {};
class InterpolateLayerTest_NPU5020 : public InterpolateLayerTestCommon {};

class InterpolateSELayerTest_NPU3720 : public InterpolateLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-se-ptrs-operations=true";
    }
};

class InterpolateLayerTest_SCFTiling : public InterpolateLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "scf-tiling=true";
        // E-190336 for MC support
        configuration["NPU_TILES"] = "1";
    }
};

TEST_P(InterpolateLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(InterpolateSELayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(InterpolateLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(InterpolateLayerTest_NPU5010, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(InterpolateLayerTest_NPU5020, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

TEST_P(InterpolateLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(InterpolateLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(InterpolateLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(InterpolateLayerTestCommon, NPU5020_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5020);
}

// SCF Tiling tests
TEST_P(InterpolateLayerTest_SCFTiling, NPU5010_SW) {
    rel_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}

TEST_P(InterpolateLayerTest_SCFTiling, NPU5020_SW) {
    rel_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> inShapes = {
        {{1, 10, 30, 30}},
};

const std::vector<ov::Shape> targetShapes = {
        {40, 40},
};

const std::vector<InterpolateBase::InterpolateMode> modesWithoutNearest = {
        InterpolateBase::InterpolateMode::LINEAR,
        InterpolateBase::InterpolateMode::LINEAR_ONNX,
        InterpolateBase::InterpolateMode::CUBIC,
};

const std::vector<InterpolateBase::InterpolateMode> nearestMode = {
        InterpolateBase::InterpolateMode::NEAREST,
};

const std::vector<InterpolateBase::InterpolateMode> linearModes = {
        InterpolateBase::InterpolateMode::LINEAR,
        InterpolateBase::InterpolateMode::LINEAR_ONNX,
};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModesNearest = {
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModeAsymmetric = {
        InterpolateBase::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModesWithoutNearest = {
        InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS,
};

const std::vector<InterpolateBase::NearestMode> nearestModes = {
        InterpolateBase::NearestMode::SIMPLE,
        InterpolateBase::NearestMode::ROUND_PREFER_FLOOR,
        InterpolateBase::NearestMode::FLOOR,
        InterpolateBase::NearestMode::CEIL,
        InterpolateBase::NearestMode::ROUND_PREFER_CEIL,
};

const std::vector<InterpolateBase::NearestMode> defaultNearestMode = {
        InterpolateBase::NearestMode::ROUND_PREFER_FLOOR,
};

const std::vector<InterpolateBase::NearestMode> defaultNearestModeFloor = {
        InterpolateBase::NearestMode::FLOOR,
};

const std::vector<std::vector<size_t>> pads = {
        // {0, 0, 1, 1},
        {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
        // Not enabled in Inference Engine
        //        true,
        false,
};

const std::vector<double> cubeCoefs = {
        -0.75f,
};

const std::vector<std::vector<int64_t>> nhwcAxes = {{1, 2}};
const std::vector<std::vector<int64_t>> nchwAxes = {{2, 3}};

const std::vector<std::vector<float>> defaultScales = {{1.33333f, 1.33333f}};
const std::vector<std::vector<float>> defaultScales2 = {{1.6666666269302368f, 1.6666666269302368f}};

const std::vector<std::vector<int64_t>> allAxes = {{0, 1, 2, 3}};
const std::vector<std::vector<float>> allScales = {{1.f, 1.f, 1.33333f, 1.33333f}};
const std::vector<ov::Shape> allScalescTargetShapes = {
        {1, 10, 40, 40},
};

const std::vector<InterpolateBase::ShapeCalcMode> shapeCalculationMode = {
        InterpolateBase::ShapeCalcMode::SIZES,
        // InterpolateBase::ShapeCalcMode::SCALES,
};

std::map<std::string, std::string> additional_config = {};

auto interpolateCasesNearestMode = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesLinearOnnxMode = [](auto scales) {
    return ::testing::Combine(::testing::Values(linearModes[1]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesWithoutNearestMode = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(modesWithoutNearest), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

auto interpolateCasesAllAxes = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(allAxes), ::testing::ValuesIn(scales));
};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_nearest_mode, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesNearestMode(defaultScales), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                            ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_nearest_mode, InterpolateLayerTest_NPU4000,
                         ::testing::Combine(interpolateCasesNearestMode(defaultScales), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                            ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_without_nearest, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesWithoutNearestMode(defaultScales),
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                            ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_without_nearest, InterpolateLayerTest_NPU4000,
                         ::testing::Combine(interpolateCasesWithoutNearestMode(defaultScales),
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                            ::testing::ValuesIn(targetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_all_axes, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCasesAllAxes(allScales), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                            ::testing::ValuesIn(allScalescTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inShapesForTiling = {
        {{1, 32, 32, 64}},
};

const std::vector<ov::Shape> targetShapesForTiling = {
        {32, 64},    // x1.00
        {128, 256},  // x4.00
                     // {136, 272}, // x4.25
                     // {144, 288}, // x4.50
                     // {152, 304}, // x4.75
};

auto makeScales = [](float uniformScale) {
    const std::vector<std::vector<float>> scales = {{uniformScale, uniformScale}};
    return scales;
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_with_tiling, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesNearestMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesForTiling)),
                           ::testing::ValuesIn(targetShapesForTiling), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_with_tiling_2x, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesNearestMode(makeScales(2.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{
                                   {{1, 3, 160, 160}}})),
                           ::testing::Values(ov::Shape{320, 320}), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_with_align_corners_tiling, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesForTiling)),
                           ::testing::ValuesIn(targetShapesForTiling), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_with_align_corners_tiling_reduce_size, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{
                                   {{1, 1, 257, 257}}})),
                           ::testing::Values(ov::Shape{17, 17}), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_with_tiling, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesNearestMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesForTiling)),
                           ::testing::ValuesIn(targetShapesForTiling), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_with_align_corners_2x, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{
                                   {{1, 512, 7, 7}}})),
                           ::testing::Values(ov::Shape{14, 14}), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_with_align_corners_2x, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{
                                   {{1, 512, 7, 7}}})),
                           ::testing::Values(ov::Shape{14, 14}), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

// test different channels
const std::vector<std::vector<ov::Shape>> inShapesForNHWCLayoutOptimize = {
        /*{{1, 1, 32, 32}},*/ {{1, 2, 32, 32}},
        {{1, 3, 32, 32}},
        {{1, 4, 32, 32}},
        {{1, 5, 32, 32}},
        {{1, 6, 32, 32}},
        {{1, 7, 32, 32}},
        {{1, 8, 32, 32}},
};

const std::vector<ov::Shape> outShapesForNHWCLayoutOptimizeSmokePrecommit = {
        {64, 64},
};

// test different output shapes
const std::vector<ov::Shape> outShapesForNHWCLayoutOptimizeSmoke = {
        {16, 16}, {24, 24}, {32, 32}, {48, 48}, {64, 64},
};

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_NHWCLayout_optimize, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesForNHWCLayoutOptimize)),
                           ::testing::ValuesIn(outShapesForNHWCLayoutOptimizeSmokePrecommit),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_NHWCLayout_optimize, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesLinearOnnxMode(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesForNHWCLayoutOptimize)),
                           ::testing::ValuesIn(outShapesForNHWCLayoutOptimizeSmoke),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

const std::vector<ov::AxisSet> axes = {{2, 3}};

const std::vector<std::vector<ov::Shape>> inShapesLargeNHWC = {
        {{1, 112, 112, 3}},
};
const std::vector<std::vector<ov::Shape>> inShapesLargeNCHW = {
        {{1, 3, 112, 112}},
};

const std::vector<ov::Shape> targetShapesLarge = {
        {150, 150},
};

const std::vector<std::vector<ov::Shape>> linearInShapesLargeNHWC = {
        {{1, 80, 80, 3}},
};
const std::vector<std::vector<ov::Shape>> linearInShapesLargeNCHW = {
        {{1, 3, 80, 80}},
};

const std::vector<ov::Shape> linearTargetShapesLarge = {
        {120, 120},
};

const std::vector<InterpolateBase::InterpolateMode> interpolateMode = {InterpolateBase::InterpolateMode::CUBIC};

auto interpolateCasesWithoutNearestModeLargerNHWC = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(interpolateMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(scales));
};
auto interpolateCasesWithoutNearestModeLargerNCHW = [](auto scales) {
    return ::testing::Combine(::testing::ValuesIn(interpolateMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

// test case for fixing input NCHW layout axes=2,3 incorrect result issue
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NCHWinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNCHW(defaultScales), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNCHW)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NCHWinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNCHW(defaultScales), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNCHW)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

// test case for input NCHW layout axes=1,2 support
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NHWCinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NHWCinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);
// test case for input NHWC layout axes=1,2 support
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NHWCinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_without_nearest_NHWCinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestModeLargerNHWC(defaultScales), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

// test case for 2D or 3D input
const std::vector<InterpolateBase::ShapeCalcMode> shapeCalculationModeSizeScale = {
        // InterpolateBase::ShapeCalcMode::SIZES,
        InterpolateBase::ShapeCalcMode::SCALES,
};

const std::vector<std::vector<size_t>> pads3D = {
        {0, 0, 0},
};

const std::vector<std::vector<ov::Shape>> inShapes3D = {
        {{8, 64, 2}},
};

const std::vector<std::vector<float>> scales3D = {
        {1.0f, 1.0f, 2.0f},
};

const std::vector<ov::Shape> targetShapes3D = {
        {8, 64, 4},
};

const std::vector<std::vector<int64_t>> AxesInput3D = {
        {0, 1, 2},
};

auto interpolateCaseNearestModeNC_Input3D = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads3D), ::testing::ValuesIn(pads3D), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(AxesInput3D), ::testing::ValuesIn(scales3D));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input3D, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCaseNearestModeNC_Input3D(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes3D)),
                                            ::testing::ValuesIn(targetShapes3D),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

// [Tracking number: E#93574]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input3D,
                         InterpolateLayerTest_NPU4000,
                         ::testing::Combine(interpolateCaseNearestModeNC_Input3D(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes3D)),
                                            ::testing::ValuesIn(targetShapes3D),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU4000::getTestCaseName);

const std::vector<std::vector<size_t>> pads2D = {
        {0, 0},
};

const std::vector<std::vector<ov::Shape>> inShapes2D = {
        {{64, 2}},
};

const std::vector<std::vector<float>> scales2D = {
        {1.0f, 2.0f},
};

const std::vector<ov::Shape> targetShapes2D = {
        {64, 4},
};

const std::vector<std::vector<int64_t>> AxesInput2D = {
        {0, 1},
};

auto interpolateCaseNearestModeNC_Input2D = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads2D), ::testing::ValuesIn(pads2D), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(AxesInput2D), ::testing::ValuesIn(scales2D));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input2D, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateCaseNearestModeNC_Input2D(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes2D)),
                                            ::testing::ValuesIn(targetShapes2D),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_nearest_NCinput_NClayout_NCaxes_Input2D, InterpolateLayerTest_NPU4000,
                         ::testing::Combine(interpolateCaseNearestModeNC_Input2D(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes2D)),
                                            ::testing::ValuesIn(targetShapes2D),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU4000::getTestCaseName);

// NEAREST cases | Axes=1,2 | Layout: NCHW and NHWC
auto interpolateCasesNearestModeAxes12 = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesNearest), ::testing::ValuesIn(nearestModes),
                              ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
                              ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes),
                              ::testing::ValuesIn(defaultScales));
};
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_nearest_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_nearest_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_nearest_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_nearest_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesNearestModeAxes12(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesLargeNHWC)),
                           ::testing::ValuesIn(targetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

const std::vector<InterpolateBase::InterpolateMode> modePytorchHalfPixel = {
        InterpolateBase::InterpolateMode::LINEAR,
        InterpolateBase::InterpolateMode::LINEAR_ONNX,
};
const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModePytorchHalfPixel = {
        InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
};
auto interpolateParamsPytorchHalfPixel = []() {
    return ::testing::Combine(::testing::ValuesIn(modePytorchHalfPixel), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModePytorchHalfPixel),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));
};
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_PytorchHalfPixel_Tiling_Upscale, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateParamsPytorchHalfPixel(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 32, 68, 120}}}))),
                                            ::testing::Values(ov::Shape{136, 240}),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_PytorchHalfPixel_Tiling_Downscale, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(interpolateParamsPytorchHalfPixel(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 3, 270, 480}}}))),
                                            ::testing::Values(ov::Shape{135, 240}),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

//
// SE Interpolate
//

const std::vector<std::vector<ov::Shape>> seInterpolateInputShapes = {
        {{1, 48, 15, 15}},
};

// DW Conv friendly shapes (in DEPTHWISE_WORKLOAD_SIZES) Scale 2x2 + HALF_PIXEL -> kernel 3x3, stride 1x1 (supports
// L1aOpt)
const std::vector<std::vector<ov::Shape>> seInterpolateDWConvInputShapes = {
        {{1, 32, 10, 10}},
};

const std::vector<std::vector<float>> seInterpolateScalesForScalesCalcMode = {
        {9.0f, 10.0f},
};

const std::vector<std::vector<float>> seInterpolateScalesForSizesCalcMode = {
        {},
};

const std::vector<ov::Shape> seInterpolateTargetShapesForScalesCalcMode = {
        {},
};

const std::vector<ov::Shape> seInterpolateTargetShapesForSizesCalcMode = {
        {127, 141},  // (127 - 1) / (15 - 1) = 9; (141 - 1) / (15 - 1) = 10
};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModesNearestSE = {
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::ASYMMETRIC,
        InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN};

auto seInterpolateParamsNearest = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModesNearestSE), ::testing::ValuesIn(nearestModes),
                              ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
                              ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes),
                              ::testing::ValuesIn(seInterpolateScalesForScalesCalcMode));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_Nearest, InterpolateSELayerTest_NPU3720,
        ::testing::Combine(seInterpolateParamsNearest(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(seInterpolateInputShapes)),
                           ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateSELayerTest_NPU3720::getTestCaseName);

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModesLinearSE = {
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::ASYMMETRIC};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformAlignCorners = {
        InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS,
};

auto seInterpolateParamsLinear = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes),
                              ::testing::Values(InterpolateBase::ShapeCalcMode::SCALES),
                              ::testing::ValuesIn(coordinateTransformModesLinearSE),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesForScalesCalcMode));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_Linear, InterpolateSELayerTest_NPU3720,
        ::testing::Combine(seInterpolateParamsLinear(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(seInterpolateInputShapes)),
                           ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateSELayerTest_NPU3720::getTestCaseName);

auto seInterpolateParamsLinearWithAlignCorners = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes),
                              ::testing::Values(InterpolateBase::ShapeCalcMode::SIZES),
                              ::testing::ValuesIn(coordinateTransformAlignCorners),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesForSizesCalcMode));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_Linear_Align_Corners, InterpolateSELayerTest_NPU3720,
        ::testing::Combine(seInterpolateParamsLinearWithAlignCorners(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(seInterpolateInputShapes)),
                           ::testing::ValuesIn(seInterpolateTargetShapesForSizesCalcMode),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateSELayerTest_NPU3720::getTestCaseName);

const std::vector<std::vector<float>> seInterpolateScalesElf = {{2.0f, 2.0f}};

auto seInterpolateParamsNearestElf = []() {
    return ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesElf));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_Nearest, InterpolateSELayerTest_NPU3720,
        ::testing::Combine(seInterpolateParamsNearestElf(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(seInterpolateInputShapes)),
                           ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateSELayerTest_NPU3720::getTestCaseName);

auto seInterpolateParamsLinearElf = []() {
    return ::testing::Combine(::testing::Values(linearModes[1]), ::testing::ValuesIn(shapeCalculationModeSizeScale),
                              ::testing::ValuesIn(coordinateTransformModeAsymmetric),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(seInterpolateScalesElf));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Interpolate_Linear, InterpolateSELayerTest_NPU3720,
        ::testing::Combine(seInterpolateParamsLinearElf(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(seInterpolateInputShapes)),
                           ::testing::ValuesIn(seInterpolateTargetShapesForScalesCalcMode),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateSELayerTest_NPU3720::getTestCaseName);

//
// Interpolate linear mode
//
auto interpolateCasesLinearOnnxModeAsymmtric = [](auto scales) {
    return ::testing::Combine(
            ::testing::Values(InterpolateBase::InterpolateMode::LINEAR_ONNX), ::testing::ValuesIn(shapeCalculationMode),
            ::testing::ValuesIn(coordinateTransformModeAsymmetric), ::testing::ValuesIn(defaultNearestMode),
            ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
            ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

const auto interpolateCasesLinearOnnxModeAsymmtricInstantiateParamsW1H2 = ::testing::Combine(
        interpolateCasesLinearOnnxModeAsymmtric(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 2, 1}}}))),
        ::testing::Values(ov::Shape{3, 1}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear_Asymmetric_W1H2, InterpolateLayerTest_NPU3720,
                         interpolateCasesLinearOnnxModeAsymmtricInstantiateParamsW1H2,
                         InterpolateLayerTest_NPU3720::getTestCaseName);

const auto interpolateCasesLinearOnnxModeAsymmtricInstantiateParamsW1H6 = ::testing::Combine(
        interpolateCasesLinearOnnxModeAsymmtric(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 6, 1}}}))),
        ::testing::Values(ov::Shape{9, 1}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear_Asymmetric_W1H6, InterpolateLayerTest_NPU3720,
                         interpolateCasesLinearOnnxModeAsymmtricInstantiateParamsW1H6,
                         InterpolateLayerTest_NPU3720::getTestCaseName);

const auto interpolateCasesLinearOnnxModeAsymmtricInstantiateParams2x = ::testing::Combine(
        interpolateCasesLinearOnnxModeAsymmtric(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 8, 16}}}))),
        ::testing::Values(ov::Shape{16, 32}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear_Asymmetric_2x, InterpolateLayerTestCommon,
                         interpolateCasesLinearOnnxModeAsymmtricInstantiateParams2x,
                         InterpolateLayerTestCommon::getTestCaseName);

const auto interpolateCasesLinearOnnxModeAsymmtricInstantiateParams4x = ::testing::Combine(
        interpolateCasesLinearOnnxModeAsymmtric(makeScales(1.f)), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 8, 16}}}))),
        ::testing::Values(ov::Shape{32, 64}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Linear_Asymmetric_4x, InterpolateLayerTestCommon,
                         interpolateCasesLinearOnnxModeAsymmtricInstantiateParams4x,
                         InterpolateLayerTestCommon::getTestCaseName);

auto interpolateCasesWithoutNearestLinearModeLargerNHWC = [](auto scales) {
    return ::testing::Combine(::testing::Values(linearModes[0]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(scales));
};
auto interpolateCasesWithoutNearestLinearModeLargerNCHW = [](auto scales) {
    return ::testing::Combine(::testing::Values(linearModes[0]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModesWithoutNearest),
                              ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scales));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNCHW)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNCHW)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNCHW)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNCHW)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NHWClayout_NCHWaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NHWClayout_NCHWaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNCHW)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NCHWlayout_NCHWaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NHWClayout_NCHWaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNCHW)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NCHWinput_NHWClayout_NCHWaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNCHW(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NCHWlayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU3720,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_linear_NHWCinput_NHWClayout_NHWCaxes, InterpolateLayerTest_NPU4000,
        ::testing::Combine(interpolateCasesWithoutNearestLinearModeLargerNHWC(defaultScales),
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(linearInShapesLargeNHWC)),
                           ::testing::ValuesIn(linearTargetShapesLarge), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config)),
        InterpolateLayerTest_NPU4000::getTestCaseName);

//
// Interpolate nearest asymmetric mode
//
const auto interpolateNearestAsymmetric = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeAsymmetric), ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));

const auto interpolateNearestAsymmetric2x = ::testing::Combine(
        interpolateNearestAsymmetric, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 4, 4}}}))),
        ::testing::Values(ov::Shape{8, 8}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_2x, InterpolateLayerTest_NPU3720,
                         interpolateNearestAsymmetric2x, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_2x, InterpolateLayerTest_NPU4000,
                         interpolateNearestAsymmetric2x, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_2x, InterpolateLayerTest_NPU5010,
                         interpolateNearestAsymmetric2x, InterpolateLayerTest_NPU5010::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_2x, InterpolateLayerTest_NPU5020,
                         interpolateNearestAsymmetric2x, InterpolateLayerTest_NPU5020::getTestCaseName);
const auto interpolateNearestAsymmetricWH = ::testing::Combine(
        interpolateNearestAsymmetric, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 4, 4}}}))),
        ::testing::Values(ov::Shape{8, 20}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_WH, InterpolateLayerTest_NPU3720,
                         interpolateNearestAsymmetricWH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_WH, InterpolateLayerTest_NPU4000,
                         interpolateNearestAsymmetricWH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_WH, InterpolateLayerTest_NPU5010,
                         interpolateNearestAsymmetricWH, InterpolateLayerTest_NPU5010::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Nearest_Asymmtric_WH, InterpolateLayerTest_NPU5020,
                         interpolateNearestAsymmetricWH, InterpolateLayerTest_NPU5020::getTestCaseName);
//
// Interpolate nearest align corner mode with tilling
//
const auto interpolateNearestAlignCorner = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModesWithoutNearest), ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(defaultScales));

const auto interpolateNearestTilingAlignCorner =
        ::testing::Combine(interpolateNearestAlignCorner, ::testing::ValuesIn(modelTypes),
                           ::testing::Values(static_shapes_to_test_representation({{1, 128, 170, 16}})),
                           ::testing::Values(ov::Shape{256, 340}), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_Nearest_Align_Corner, InterpolateLayerTestCommon,
                         interpolateNearestTilingAlignCorner, InterpolateLayerTestCommon::getTestCaseName);
const std::vector<std::vector<int64_t>> axesInput5D = {
        {2, 3, 4},
};
const std::vector<std::vector<float>> scales5D = {
        {2.0f, 2.0f, 2.0f},
};
const std::vector<ov::Shape> targetShapes5D = {
        {4, 4, 4},
};

const auto interpolate5D = ::testing::Combine(
        ::testing::Values(ov::op::v4::Interpolate::InterpolateMode::NEAREST), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeAsymmetric), ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesInput5D), ::testing::ValuesIn(scales5D));
const auto interpolate5D_3axes =
        ::testing::Combine(interpolate5D, ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 2, 2, 2, 2}}, {{2, 2, 2, 2, 2}}}))),
                           ::testing::ValuesIn(targetShapes5D), ::testing::Values(test_utils::TARGET_DEVICE),
                           ::testing::Values(additional_config));
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_3axes_5D, InterpolateLayerTest_NPU3720, interpolate5D_3axes,
                         InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_3axes_5D, InterpolateLayerTest_NPU4000, interpolate5D_3axes,
                         InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_3axes_5D, InterpolateLayerTest_NPU5010, interpolate5D_3axes,
                         InterpolateLayerTest_NPU5010::getTestCaseName);

// --------------------------------------------------
// ------ Common NoTiling Interpolate Testing (all platforms) ------
// --------------------------------------------------

const std::vector<std::vector<int64_t>> axesComplete = {{1, 2}, {2, 3}};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModeComplete = {
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN,
        InterpolateBase::CoordinateTransformMode::ASYMMETRIC,
        InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS,
};

const auto interpolateCasesNearestModeComplete = ::testing::Combine(
        ::testing::Values(InterpolateBase::InterpolateMode::NEAREST), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(nearestModes),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesComplete), ::testing::ValuesIn(defaultScales));
const auto interpolateParamsLinear = ::testing::Combine(
        ::testing::Values(InterpolateBase::InterpolateMode::LINEAR), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesComplete), ::testing::ValuesIn(defaultScales));

const auto interpolateParamsCubic = ::testing::Combine(
        ::testing::Values(InterpolateBase::InterpolateMode::CUBIC), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(axesComplete), ::testing::ValuesIn(defaultScales));

const auto interpolateParamsLinearONNX = ::testing::Combine(
        ::testing::Values(InterpolateBase::InterpolateMode::LINEAR_ONNX), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));

const auto interpolateNearestNCHWUpscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNearestNHWCUpscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNearestNCHWDownscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNearestNHWCDownscale = ::testing::Combine(
        interpolateCasesNearestModeComplete, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

const auto interpolateLinearNCHWUpscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNHWCUpscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNCHWDownscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNHWCDownscale = ::testing::Combine(
        interpolateParamsLinear, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

const auto interpolateLinearONNXNCHWUpscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearONNXNHWCUpscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearONNXNCHWDownscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearONNXNHWCDownscale = ::testing::Combine(
        interpolateParamsLinearONNX, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

const auto interpolateCubicNCHWUpscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateCubicNHWCUpscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 30, 30}}}))),
        ::testing::Values(ov::Shape{40, 40}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateCubicNCHWDownscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateCubicNHWCDownscale = ::testing::Combine(
        interpolateParamsCubic, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 40, 40}}}))),
        ::testing::Values(ov::Shape{30, 30}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// Mode NEAREST | Axes {1,2} & {2,3} | Coord Transform Mode: ALL | Nearest Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NCHW_Upscale, InterpolateLayerTestCommon,
                         interpolateNearestNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NHWC_Upscale, InterpolateLayerTestCommon,
                         interpolateNearestNHWCUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NCHW_Downscale, InterpolateLayerTestCommon,
                         interpolateNearestNCHWDownscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Nearest_NHWC_Downscale, InterpolateLayerTestCommon,
                         interpolateNearestNHWCDownscale, InterpolateLayerTestCommon::getTestCaseName);

// Mode LINEAR | Axes {1,2} & {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NCHW_Upscale, InterpolateLayerTestCommon,
                         interpolateLinearNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NHWC_Upscale, InterpolateLayerTestCommon,
                         interpolateLinearNHWCUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NCHW_Downscale, InterpolateLayerTestCommon,
                         interpolateLinearNCHWDownscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Linear_NHWC_Downscale, InterpolateLayerTestCommon,
                         interpolateLinearNHWCDownscale, InterpolateLayerTestCommon::getTestCaseName);

// Mode LINEAR_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NCHW_Upscale, InterpolateLayerTestCommon,
                         interpolateLinearONNXNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NHWC_Upscale, InterpolateLayerTestCommon,
                         interpolateLinearONNXNHWCUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NCHW_Downscale, InterpolateLayerTestCommon,
                         interpolateLinearONNXNCHWDownscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_LinearONNX_NHWC_Downscale, InterpolateLayerTestCommon,
                         interpolateLinearONNXNHWCDownscale, InterpolateLayerTestCommon::getTestCaseName);

// Mode CUBIC | Axes {1,2} & {2,3} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NCHW_Upscale, InterpolateLayerTestCommon,
                         interpolateCubicNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NHWC_Upscale, InterpolateLayerTestCommon,
                         interpolateCubicNHWCUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NCHW_Downscale, InterpolateLayerTestCommon,
                         interpolateCubicNCHWDownscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_NoTiling_Cubic_NHWC_Downscale, InterpolateLayerTestCommon,
                         interpolateCubicNHWCDownscale, InterpolateLayerTestCommon::getTestCaseName);

// NoTiling Precommit (all platforms)
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_NoTiling_Nearest, InterpolateLayerTestCommon,
                         interpolateNearestNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_NoTiling_Linear, InterpolateLayerTestCommon,
                         interpolateLinearNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_NoTiling_LinearONNX, InterpolateLayerTestCommon,
                         interpolateLinearONNXNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Interpolate_NoTiling_Cubic, InterpolateLayerTestCommon,
                         interpolateCubicNCHWUpscale, InterpolateLayerTestCommon::getTestCaseName);

//
// Optimize bilinear Interpolate with HALF_PIXEL and PYTORCH_HALF_PIXEL modes through the conversion to
// (depth-)convolution and DMA
//

const std::vector<std::vector<ov::Shape>> bilinearInterpolateInputShapes = {{{1, 40, 40, 40}, {1, 32, 40, 40}}};
const std::vector<std::vector<ov::Shape>> interpolateAccuracyIssueInputShapes = {{{1, 3, 4, 6}}};
const std::vector<std::vector<ov::Shape>> interpolateOutputShapeInferenceInputShapes = {{{1, 3, 3, 3}}};

const std::vector<ov::Shape> bilinearInterpolateTargetShapes = {
        {80, 80}, {120, 120}, {160, 160}, {240, 240}, {80, 120}};
const std::vector<ov::Shape> interpolateEmptyTargetShapes = {{}};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModeHalfPixelandPytorchHalfPixel = {
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
};

auto bilinearInterpolateParamsLinear = []() {
    return ::testing::Combine(::testing::ValuesIn(linearModes), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModeHalfPixelandPytorchHalfPixel),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));
};

auto bilinearInterpolateAccuracy = []() {
    return ::testing::Combine(::testing::Values(InterpolateBase::InterpolateMode::LINEAR),
                              ::testing::Values(InterpolateBase::ShapeCalcMode::SCALES),
                              ::testing::Values(InterpolateBase::CoordinateTransformMode::HALF_PIXEL),
                              ::testing::Values(InterpolateBase::NearestMode::FLOOR), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));
};

auto bilinearInterpolateOutputShapeInference = []() {
    return ::testing::Combine(::testing::Values(InterpolateBase::InterpolateMode::LINEAR),
                              ::testing::Values(InterpolateBase::ShapeCalcMode::SCALES),
                              ::testing::Values(InterpolateBase::CoordinateTransformMode::HALF_PIXEL),
                              ::testing::Values(InterpolateBase::NearestMode::FLOOR), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales2));
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Interpolate_bilinearInterpolateToConv, InterpolateLayerTestCommon,
        ::testing::Combine(bilinearInterpolateParamsLinear(), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(bilinearInterpolateInputShapes)),
                           ::testing::ValuesIn(bilinearInterpolateTargetShapes),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config)),
        InterpolateLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_Accuracy, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(bilinearInterpolateAccuracy(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    interpolateAccuracyIssueInputShapes)),
                                            ::testing::ValuesIn(interpolateEmptyTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_OutputShape_Inference, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(bilinearInterpolateOutputShapeInference(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    interpolateOutputShapeInferenceInputShapes)),
                                            ::testing::ValuesIn(interpolateEmptyTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_Accuracy, InterpolateLayerTest_NPU4000,
                         ::testing::Combine(bilinearInterpolateAccuracy(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    interpolateAccuracyIssueInputShapes)),
                                            ::testing::ValuesIn(interpolateEmptyTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_OutputShape_Inference, InterpolateLayerTest_NPU4000,
                         ::testing::Combine(bilinearInterpolateOutputShapeInference(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    interpolateOutputShapeInferenceInputShapes)),
                                            ::testing::ValuesIn(interpolateEmptyTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_Accuracy, InterpolateLayerTest_NPU5010,
                         ::testing::Combine(bilinearInterpolateAccuracy(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    interpolateAccuracyIssueInputShapes)),
                                            ::testing::ValuesIn(interpolateEmptyTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU5010::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_OutputShape_Inference, InterpolateLayerTest_NPU5010,
                         ::testing::Combine(bilinearInterpolateOutputShapeInference(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    interpolateOutputShapeInferenceInputShapes)),
                                            ::testing::ValuesIn(interpolateEmptyTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU5010::getTestCaseName);

//
// SCALES mode with non-integer scale where floor(input*scale) != input*scale
//
// Tests the semantic divergence between SCALES and SIZES modes under tiling.
// Input 150x150, scale=1.6133 → exact output = 241.995, floor = 241.
// The isTiled fallback in backInferOffsetForInterpolate recomputes the backward
// scale as the dim ratio 150/241 = 0.62241 (≡ effective scale 1.6067) instead
// of using 1/originalScales = 1/1.6133 = 0.61985.
// This 0.00256 difference flips floor() at cluster-boundary output rows within
// the second tile, leaving a one-row hole in each affected cluster's input
// distribution and producing ~1.245% wrong output pixels (worst |Δ| ≈ 7.56)
// when the SHAVE kernel falls back to its bounds-clamp on the missing row.
//

const std::vector<std::vector<ov::Shape>> inShapesScalesDivergence = {
        {{1, 32, 150, 150}},
};

// scale=1.6133: 150 * 1.6133 = 241.995, floor = 241 (gap = 0.995)
// Per-cluster (3 NCE, SplitOverH): ~1.66 MB > 1.41 MB CMX → forces spatial tiling in DefaultHW
// After tiling [1,1,2,1]: recomputed scales diverge from 1.6133 by up to 0.042
const std::vector<std::vector<float>> scalesScalesDivergence = {{1.6133f, 1.6133f}};

const std::vector<InterpolateBase::ShapeCalcMode> shapeCalculationModeScalesOnly = {
        InterpolateBase::ShapeCalcMode::SCALES,
};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModesScalesDivergence = {
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
        InterpolateBase::CoordinateTransformMode::ASYMMETRIC,
};

const std::vector<InterpolateBase::NearestMode> nearestModesScalesDivergence = {
        InterpolateBase::NearestMode::FLOOR,
};

auto interpolateScalesDivergenceNearest =
        ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeScalesOnly),
                           ::testing::ValuesIn(coordinateTransformModesScalesDivergence),
                           ::testing::ValuesIn(nearestModesScalesDivergence), ::testing::ValuesIn(antialias),
                           ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                           ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scalesScalesDivergence));

// target_shape is unused when shape_calc_mode=SCALES; output is derived from floor(input * scale)
auto interpolateScalesDivergenceParams =
        ::testing::Combine(interpolateScalesDivergenceNearest, ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesScalesDivergence)),
                           ::testing::ValuesIn(interpolateEmptyTargetShapes),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Scale_Accuracy_NonIntegerScale, InterpolateLayerTestCommon,
                         interpolateScalesDivergenceParams, InterpolateLayerTestCommon::getTestCaseName);

//
// MapInterpolateOnDPU
//

const std::vector<std::vector<float>> mapBilinearInterpolateOnDPUScales = {{1.9444544315338135, 1.9444544315338135}};

const std::vector<std::vector<ov::Shape>> mapBilinearInterpolateOnDPUInputShapes = {
        {{1, 80, 72, 72}},
};

const std::vector<ov::Shape> mapBilinearInterpolateOnDPUTargetShapes = {
        {1, 80, 140, 140},
};

auto mapBilinearInterpolateOnDPUParamsLinear = []() {
    return ::testing::Combine(::testing::Values(linearModes[1]), ::testing::ValuesIn(shapeCalculationMode),
                              ::testing::ValuesIn(coordinateTransformModeComplete),
                              ::testing::ValuesIn(defaultNearestModeFloor), ::testing::ValuesIn(antialias),
                              ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs),
                              ::testing::ValuesIn(allAxes), ::testing::ValuesIn(mapBilinearInterpolateOnDPUScales));
};

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_MapBilinearInterpolateOnDPU, InterpolateLayerTest_NPU3720,
                         ::testing::Combine(mapBilinearInterpolateOnDPUParamsLinear(), ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    mapBilinearInterpolateOnDPUInputShapes)),
                                            ::testing::ValuesIn(mapBilinearInterpolateOnDPUTargetShapes),
                                            ::testing::Values(test_utils::TARGET_DEVICE),
                                            ::testing::Values(additional_config)),
                         InterpolateLayerTest_NPU3720::getTestCaseName);

// --------------------------------------------------
// ------ NPU3720/NPU4000 Tiling Interpolate Testing ------
// --------------------------------------------------

const std::vector<InterpolateBase::InterpolateMode> interpolateAxes12ModeComplete = {
        InterpolateBase::InterpolateMode::LINEAR,
};

const auto interpolateParamsAxes12 = ::testing::Combine(
        ::testing::ValuesIn(interpolateAxes12ModeComplete), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nhwcAxes), ::testing::ValuesIn(defaultScales));
const auto interpolateParamsAxes23 = ::testing::Combine(
        ::testing::ValuesIn(linearModes), ::testing::ValuesIn(shapeCalculationMode),
        ::testing::ValuesIn(coordinateTransformModeComplete), ::testing::ValuesIn(defaultNearestMode),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));

// UpScale| Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW
// and NHWC
const auto interpolateNCHWUpscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 4, 100, 180}}}))),
        ::testing::Values(ov::Shape{440, 550}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNCHWUpscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 460, 620}}}))),
        ::testing::Values(ov::Shape{800, 1000}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNHWCUpscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 4, 99, 181}}}))),
        ::testing::Values(ov::Shape{440, 550}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNHWCUpscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 190, 580}}}))),
        ::testing::Values(ov::Shape{500, 750}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// UpScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
const auto interpolateLinearNCHWUpscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 3, 127, 540}}}))),
        ::testing::Values(ov::Shape{5, 317}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNCHWUpscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 3, 160, 520}}}))),
        ::testing::Values(ov::Shape{5, 300}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNHWCUpscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 131, 630}}}))),
        ::testing::Values(ov::Shape{4, 317}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNHWCUpscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 230, 400}}}))),
        ::testing::Values(ov::Shape{4, 500}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// DownScale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts:
// NCHW and NHWC
const auto interpolateNCHWDownscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 4, 336, 640}}}))),
        ::testing::Values(ov::Shape{144, 256}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNCHWDownscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 900, 700}}}))),
        ::testing::Values(ov::Shape{760, 520}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNHWCDownscaleAxes23TileC = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 4, 359, 639}}}))),
        ::testing::Values(ov::Shape{144, 256}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateNHWCDownscaleAxes23TileH = ::testing::Combine(
        interpolateParamsAxes23, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 600, 700}}}))),
        ::testing::Values(ov::Shape{230, 560}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// DownScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
const auto interpolateLinearNCHWDownscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 5, 359, 640}}}))),
        ::testing::Values(ov::Shape{3, 143}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNCHWDownscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 5, 250, 620}}}))),
        ::testing::Values(ov::Shape{3, 160}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNHWCDownscaleAxes12TileW = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 4, 359, 630}}}))),
        ::testing::Values(ov::Shape{2, 143}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));
const auto interpolateLinearNHWCDownscaleAxes12TileH = ::testing::Combine(
        interpolateParamsAxes12, ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 4, 600, 400}}}))),
        ::testing::Values(ov::Shape{2, 230}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// UpScale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts: NCHW
// and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNCHWUpscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNCHWUpscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNHWCUpscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNHWCUpscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// UpScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWUpscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWUpscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCUpscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCUpscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// DownScale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts:
// NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNCHWDownscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNCHWDownscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes23_tileC, InterpolateLayerTest_NPU3720,
                         interpolateNHWCDownscaleAxes23TileC, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes23_tileH, InterpolateLayerTest_NPU3720,
                         interpolateNHWCDownscaleAxes23TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// DownScale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWDownscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNCHWDownscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes12_tileW, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCDownscaleAxes12TileW, InterpolateLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes12_tileH, InterpolateLayerTest_NPU3720,
                         interpolateLinearNHWCDownscaleAxes12TileH, InterpolateLayerTest_NPU3720::getTestCaseName);

// --------------------------------------------------
// ------ NPU4000 Tiling Interpolate Testing ------
// --------------------------------------------------

// Upscale | Interpolate mode : Linear and Linear_ONNX | Axes  {2,3} | Coord Transform Mode: ALL | Layouts: NCHW
// and NHWC

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes23_tileC, InterpolateLayerTest_NPU4000,
                         interpolateNCHWUpscaleAxes23TileC, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes23_tileC, InterpolateLayerTest_NPU4000,
                         interpolateNHWCUpscaleAxes23TileC, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes23_tileH, InterpolateLayerTest_NPU4000,
                         interpolateNCHWUpscaleAxes23TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes23_tileH, InterpolateLayerTest_NPU4000,
                         interpolateNHWCUpscaleAxes23TileH, InterpolateLayerTest_NPU4000::getTestCaseName);

// Upscale | Interpolate mode : Linear | Axes  {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes12_tileW, InterpolateLayerTest_NPU4000,
                         interpolateLinearNCHWUpscaleAxes12TileW, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes12_tileH, InterpolateLayerTest_NPU4000,
                         interpolateLinearNHWCUpscaleAxes12TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Upscale_axes12_tileH, InterpolateLayerTest_NPU4000,
                         interpolateLinearNCHWUpscaleAxes12TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Upscale_axes12_tileW, InterpolateLayerTest_NPU4000,
                         interpolateLinearNHWCUpscaleAxes12TileW, InterpolateLayerTest_NPU4000::getTestCaseName);

// Downscale | Interpolate mode : Linear and Linear_ONNX | Axes {2,3} | Coord Transform Mode: ALL | Layouts:
// NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes23_tileH, InterpolateLayerTest_NPU4000,
                         interpolateNCHWDownscaleAxes23TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes23_tileC, InterpolateLayerTest_NPU4000,
                         interpolateNHWCDownscaleAxes23TileC, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes23_tileH, InterpolateLayerTest_NPU4000,
                         interpolateNHWCDownscaleAxes23TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes23_tileC, InterpolateLayerTest_NPU4000,
                         interpolateNCHWDownscaleAxes23TileC, InterpolateLayerTest_NPU4000::getTestCaseName);

// Downscale | Interpolate mode : Linear | Axes {1,2} | Coord Transform Mode: ALL | Layouts: NCHW and NHWC
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes12_tileW, InterpolateLayerTest_NPU4000,
                         interpolateLinearNCHWDownscaleAxes12TileW, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes12_tileH, InterpolateLayerTest_NPU4000,
                         interpolateLinearNHWCDownscaleAxes12TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NCHW_Downscale_axes12_tileH, InterpolateLayerTest_NPU4000,
                         interpolateLinearNCHWDownscaleAxes12TileH, InterpolateLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_Tiling_NHWC_Downscale_axes12_tileW, InterpolateLayerTest_NPU4000,
                         interpolateLinearNHWCDownscaleAxes12TileW, InterpolateLayerTest_NPU4000::getTestCaseName);

// ------ SCF Tiling Tests ------

const std::vector<std::vector<ov::Shape>> inShapesSCFTiling = {
        {{1, 32, 128, 128}},
};

const std::vector<ov::Shape> targetShapesSCFTilingUpscale = {
        {256, 256},  // 2x upscale
};

const std::vector<ov::Shape> targetShapesSCFTilingDownscale = {
        {64, 64},  // 0.5x downscale
};

const std::vector<std::vector<float>> scalesSCFUpscale = {{2.0f, 2.0f}};
const std::vector<std::vector<float>> scalesSCFDownscale = {{0.5f, 0.5f}};

const std::vector<InterpolateBase::ShapeCalcMode> shapeCalculationModeSCF = {
        InterpolateBase::ShapeCalcMode::SCALES,
};

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModesSCF = {
        InterpolateBase::CoordinateTransformMode::ASYMMETRIC,
        InterpolateBase::CoordinateTransformMode::HALF_PIXEL,
};

const std::vector<InterpolateBase::NearestMode> nearestModesSCF = {
        InterpolateBase::NearestMode::FLOOR,
        InterpolateBase::NearestMode::ROUND_PREFER_FLOOR,
};

auto interpolateSCFNearestUpscale = ::testing::Combine(
        ::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSCF),
        ::testing::ValuesIn(coordinateTransformModesSCF), ::testing::ValuesIn(nearestModesSCF),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scalesSCFUpscale));

auto interpolateSCFNearestDownscale = ::testing::Combine(
        ::testing::ValuesIn(nearestMode), ::testing::ValuesIn(shapeCalculationModeSCF),
        ::testing::ValuesIn(coordinateTransformModesSCF), ::testing::ValuesIn(nearestModesSCF),
        ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
        ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(scalesSCFDownscale));

auto interpolateSCFTilingNearestUpscaleParams =
        ::testing::Combine(interpolateSCFNearestUpscale, ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesSCFTiling)),
                           ::testing::ValuesIn(targetShapesSCFTilingUpscale),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config));

auto interpolateSCFTilingNearestDownscaleParams =
        ::testing::Combine(interpolateSCFNearestDownscale, ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapesSCFTiling)),
                           ::testing::ValuesIn(targetShapesSCFTilingDownscale),
                           ::testing::Values(test_utils::TARGET_DEVICE), ::testing::Values(additional_config));

// SCF Tiling - Nearest mode
INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_SCFTiling_Nearest_Upscale, InterpolateLayerTest_SCFTiling,
                         interpolateSCFTilingNearestUpscaleParams, InterpolateLayerTest_SCFTiling::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_SCFTiling_Nearest_Downscale, InterpolateLayerTest_SCFTiling,
                         interpolateSCFTilingNearestDownscaleParams, InterpolateLayerTest_SCFTiling::getTestCaseName);

//
// Extreme downscale — reproduces interpLinearCHW stack buffer overflow (R > 32)
// Bug: splitting loop cannot converge when downscale ratio exceeds stack buffer capacity
//

const std::vector<InterpolateBase::CoordinateTransformMode> coordinateTransformModePytorchHP = {
        InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL,
};

auto interpolateCasesLinearOnnxExtremeDownscale = []() {
    return ::testing::Combine(
            ::testing::Values(InterpolateBase::InterpolateMode::LINEAR_ONNX), ::testing::ValuesIn(shapeCalculationMode),
            ::testing::ValuesIn(coordinateTransformModePytorchHP), ::testing::ValuesIn(defaultNearestMode),
            ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads), ::testing::ValuesIn(pads),
            ::testing::ValuesIn(cubeCoefs), ::testing::ValuesIn(nchwAxes), ::testing::ValuesIn(defaultScales));
};

// 320x downscale on W axis
const auto interpolateExtremeDownscale320x = ::testing::Combine(
        interpolateCasesLinearOnnxExtremeDownscale(), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 4, 3200}}}))),
        ::testing::Values(ov::Shape{4, 10}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// 160x downscale on W axis with multiple channels
const auto interpolateExtremeDownscale160x = ::testing::Combine(
        interpolateCasesLinearOnnxExtremeDownscale(), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 2, 4, 1600}}}))),
        ::testing::Values(ov::Shape{4, 10}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// 64x downscale — just above the R>32 threshold for NPU4000
const auto interpolateExtremeDownscale64x = ::testing::Combine(
        interpolateCasesLinearOnnxExtremeDownscale(), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 4, 640}}}))),
        ::testing::Values(ov::Shape{4, 10}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

// 34x downscale — boundary case just above R=32 threshold
const auto interpolateExtremeDownscale34x = ::testing::Combine(
        interpolateCasesLinearOnnxExtremeDownscale(), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 1, 4, 340}}}))),
        ::testing::Values(ov::Shape{4, 10}), ::testing::Values(test_utils::TARGET_DEVICE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_LinearONNX_ExtremeDownscale_320x, InterpolateLayerTestCommon,
                         interpolateExtremeDownscale320x, InterpolateLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_LinearONNX_ExtremeDownscale_160x, InterpolateLayerTestCommon,
                         interpolateExtremeDownscale160x, InterpolateLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_LinearONNX_ExtremeDownscale_64x, InterpolateLayerTestCommon,
                         interpolateExtremeDownscale64x, InterpolateLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_LinearONNX_ExtremeDownscale_34x, InterpolateLayerTestCommon,
                         interpolateExtremeDownscale34x, InterpolateLayerTestCommon::getTestCaseName);

}  // namespace
