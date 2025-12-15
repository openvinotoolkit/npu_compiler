//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/pad.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class Pad12LayerTestCommon : public Pad12LayerTest, virtual public VpuOv2LayerTest {};
class PadLayerTestCommon : public PadLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {  // allow both f16/f32 tests
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

class PadLayerTestSW : public PadLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {  // don't allow mapping to DPU
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-convert-non-constant-pad-to-slice-and-concat=false";
    }
};

TEST_P(PadLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(PadLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
TEST_P(PadLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}
TEST_P(Pad12LayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(Pad12LayerTestCommon, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(Pad12LayerTestCommon, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
TEST_P(PadLayerTestSW, NPU4000_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(PadLayerTestSW, NPU5010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

/* ================================= Pad arch >= NPU3720 ================================= */
// Note (subject to change):
// - most common padding modes (seen in PowerBI): CONSTANT & REFLECT map on DMA (for f16|f32, SW and HW pipeline)

const std::vector<std::vector<int64_t>> padsBegin = {{0, 0, 0, 0}, {0, 3, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd = {{0, 0, 0, 0}, {0, 3, 2, 4}};
const std::vector<ov::op::PadMode> padModes = {ov::op::PadMode::CONSTANT, ov::op::PadMode::EDGE,
                                               ov::op::PadMode::REFLECT, ov::op::PadMode::SYMMETRIC};
const auto pad4DParams = testing::Combine(
        testing::ValuesIn(padsBegin), testing::ValuesIn(padsEnd), testing::Values(0.0f), testing::ValuesIn(padModes),
        testing::Values(ov::element::f16),
        testing::ValuesIn(static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{{{1, 5, 10, 11}}})),
        testing::Values(test_utils::TARGET_DEVICE));

// returns a single explicit test param
padLayerTestParamsSet getCfg(ov::Shape inShape, std::vector<int64_t> padsBegin, std::vector<int64_t> padsEnd,
                             ov::op::PadMode padMode, ov::element::Type prc) {
    auto shape = static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>{{inShape}})[0];
    return std::make_tuple(padsBegin, padsEnd, -1.0f, padMode, prc, shape, test_utils::TARGET_DEVICE);
}

std::vector<padLayerTestParamsSet> customParams = {
        // net configs
        getCfg(ov::Shape({1, 1, 64, 256}), {0, 0, 0, 0}, {0, 0, 0, 2}, ov::op::PadMode::EDGE, ov::element::f16),
        getCfg(ov::Shape({1, 32, 128, 1}), {0, 0, 1, 0}, {0, 0, 1, 0}, ov::op::PadMode::EDGE, ov::element::f16),
};

std::vector<padLayerTestParamsSet> precommitParams = {
        getCfg(ov::Shape({1, 5, 10, 11}), {0, 0, 2, 0}, {0, 1, 0, 3}, ov::op::PadMode::REFLECT, ov::element::f32),
        getCfg(ov::Shape({1, 5, 10, 11}), {4, 2, 1, 3}, {5, 2, 6, 1}, ov::op::PadMode::CONSTANT, ov::element::f16),
        getCfg(ov::Shape({1, 5, 10, 11}), {0, 2, 1, 3}, {0, 2, 6, 1}, ov::op::PadMode::SYMMETRIC, ov::element::f16)};

std::vector<padLayerTestParamsSet> reflectPadParams = {
        getCfg(ov::Shape({1, 5, 10, 11}), {0, 0, 2, 0}, {0, 1, 0, 3}, ov::op::PadMode::REFLECT, ov::element::f16),
        getCfg(ov::Shape({1, 2, 1, 16}), {0, 1, 0, 0}, {0, 0, 0, 0}, ov::op::PadMode::REFLECT, ov::element::f16),
        getCfg(ov::Shape({1, 256, 256, 3}), {0, 40, 40, 0}, {0, 40, 40, 0}, ov::op::PadMode::REFLECT, ov::element::f16),
        getCfg(ov::Shape({1, 10, 32, 16}), {0, 0, 1, 0}, {0, 0, 15, 0}, ov::op::PadMode::REFLECT, ov::element::f16)};

// -------------- all PadLayerTestCommon arch instances
INSTANTIATE_TEST_SUITE_P(smoke_Pad, PadLayerTestCommon, pad4DParams, PadLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pad, PadLayerTestCommon, ::testing::ValuesIn(precommitParams),
                         PadLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(custom_Pad, PadLayerTestCommon, ::testing::ValuesIn(customParams),
                         PadLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_reflectPad, PadLayerTestSW, ::testing::ValuesIn(reflectPadParams),
                         PadLayerTestSW::getTestCaseName);

}  // namespace

namespace {  // Pad12

INSTANTIATE_TEST_SUITE_P(smoke_Pad12, Pad12LayerTestCommon,
                         testing::Combine(testing::Values(std::vector<int64_t>({0, 1, 1, 1})),
                                          testing::Values(std::vector<int64_t>({0, 1, 1, 1})), testing::Values(1.f),
                                          testing::Values(ov::op::PadMode::CONSTANT), testing::Values(ov::element::f16),
                                          testing::Values(ov::test::static_shapes_to_test_representation(
                                                  std::vector<ov::Shape>{{1, 5, 10, 11}})),
                                          testing::Values(test_utils::TARGET_DEVICE)),
                         Pad12LayerTestCommon::getTestCaseName);

}  // namespace
