//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/fake_quantize.hpp"
#include <random>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class FakeQuantizeLayerTestCommon : public FakeQuantizeLayerTest, virtual public VpuOv2LayerTest {};

class FakeQuantizeLayerTest_SW_NPU3720 : public FakeQuantizeLayerTestCommon {
    //     Use realistic float inputs (default generator produces int data)
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto& specificParams = std::get<0>(GetParam());
        const auto& limits = std::get<2>(specificParams);
        float low, high;

        if (limits.empty()) {
            low = 0;  // match 'makeFakeQuantize' default ranges
            high = 12;
        } else {
            low = limits[0];  // use user ranges
            high = limits[1];
        }
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();

        std::mt19937 gen(123);
        const float extra = 0.2f;
        std::uniform_real_distribution<float> dist(low - extra, high + extra);
        for (size_t i = 0; i < totalSize; i++) {
            auto f16 = static_cast<ov::fundamental_type_for<ov::element::f16>>(dist(gen));
            inputData[i] = f16.to_bits();
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }
};

class FakeQuantizeLayerTest_HW_NPU3720 : public FakeQuantizeLayerTestCommon {};

class FakeQuantizeLayerTest_SW_NPU4000 : public FakeQuantizeLayerTest_SW_NPU3720 {};
class FakeQuantizeLayerTest_SW_NPU5010 : public FakeQuantizeLayerTest_SW_NPU3720 {};

TEST_P(FakeQuantizeLayerTest_SW_NPU3720, SW) {
    const auto tol = 1.6;                       // To cope with cpu/npu 'limits' diffs
    rel_threshold = fabs(rel_threshold) * tol;  // E#77437
    abs_threshold = rel_threshold;              // Rely on absolute value check
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(FakeQuantizeLayerTest_HW_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FakeQuantizeLayerTest_SW_NPU4000, SW) {
    const auto tol = 1.6;                       // To cope with cpu/npu 'limits' diffs
    rel_threshold = fabs(rel_threshold) * tol;  // E#77437
    abs_threshold = rel_threshold;              // Rely on absolute value check
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}
TEST_P(FakeQuantizeLayerTest_SW_NPU5010, SW) {
    const auto tol = 1.6;                       // To cope with cpu/npu 'limits' diffs
    rel_threshold = fabs(rel_threshold) * tol;  // E#77437
    abs_threshold = rel_threshold;              // Rely on absolute value check
    setReferenceSoftwareMode();
    run(Platform::NPU5010);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> inputShapes = {{{1, 3, 10, 10}}};
const std::vector<std::vector<size_t>> constShapes = {{1}, {1, 3, 1, 1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::vector<std::vector<size_t>> constShapesND = {{1}};

const std::vector<float> fqArgs = {0, 255, 0, 255};

// Per-Tensor
const std::vector<size_t> u8qLevels = {256};

const std::vector<std::vector<ov::Shape>> inShapes3720 = {
        {{2, 3, 10, 10}},
        {{1, 32, 16, 8}},
};

const std::vector<std::vector<ov::Shape>> tilingShapes3720 = {
        {{1, 128, 64, 64}},
        {{1, 64, 128, 80}},
        {{1, 256, 80, 80}},
};

// {inLow, inHigh, outLow, outHigh}
const std::vector<std::vector<float>> fqLimits = {{+0.00, +0.90, +0.00, +0.90}, {+4.50, +9.80, +4.55, +9.74},
                                                  {-5.20, -1.50, -5.15, -1.53}, {-0.50, +0.60, +0.62, -0.58},
                                                  {-0.50, +1.60, -0.40, +1.62}, {-39.0, +231.0, -28.0, +250.0}};

const auto fqParamsU =
        ::testing::Combine(::testing::ValuesIn(u8qLevels), ::testing::Values(constShapes[0]),
                           ::testing::ValuesIn(fqLimits), ::testing::Values(ov::op::AutoBroadcastType::NUMPY));

const auto perTensorCfg = ::testing::Combine(fqParamsU, ::testing::Values(ov::element::f16),
                                             ::testing::ValuesIn(static_shapes_to_test_representation(inShapes3720)),
                                             ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerTensor, FakeQuantizeLayerTest_SW_NPU3720, perTensorCfg,
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerTensor, FakeQuantizeLayerTest_SW_NPU4000, perTensorCfg,
                         FakeQuantizeLayerTest_SW_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerTensor, FakeQuantizeLayerTest_SW_NPU5010, perTensorCfg,
                         FakeQuantizeLayerTest_SW_NPU5010::getTestCaseName);
// NPU3720 Per-Tensor Tiling
const auto fqParamsT =
        ::testing::Combine(::testing::ValuesIn(u8qLevels), ::testing::Values(constShapes[0]),
                           ::testing::Values(fqLimits[0]), ::testing::Values(ov::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(
        smoke_tiling_FakeQuantize_PerTensor, FakeQuantizeLayerTest_SW_NPU3720,
        ::testing::Combine(fqParamsT, ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn({static_shapes_to_test_representation(tilingShapes3720[2])}),
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Per-Channel (different lo/hi limits per channel)

// Helper to keep 'input' and 'limits' shapes aligned
const auto perChParams(std::vector<ov::Shape> inShape) {
    const auto levels = 255;
    const std::vector<float> noLimits = {};  // empty => per channel default inits
    std::vector<size_t> ctShape = {1, inShape[0][1], 1, 1};

    const auto fqParams =
            ::testing::Combine(::testing::Values(levels), ::testing::Values(ctShape), ::testing::Values(noLimits),
                               ::testing::Values(ov::op::AutoBroadcastType::NUMPY));

    return ::testing::Combine(fqParams, ::testing::Values(ov::element::f16),
                              ::testing::Values(static_shapes_to_test_representation(inShape)),
                              ::testing::Values(test_utils::TARGET_DEVICE));
}

INSTANTIATE_TEST_SUITE_P(smoke_precommit_FakeQuantize_PerCh_a, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<ov::Shape>{{1, 3, 10, 10}}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerCh_b, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<ov::Shape>{{1, 8, 9, 9}}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerCh_c, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<ov::Shape>{{1, 17, 5, 2}}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerCh_d, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<ov::Shape>{{1, 32, 3, 3}}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Per-Channel Tiling tests
INSTANTIATE_TEST_SUITE_P(smoke_tiling_FakeQuantize_PerCh_a, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(tilingShapes3720[0]), FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_FakeQuantize_PerCh_b, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(tilingShapes3720[1]), FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Fp32 input
const std::vector<size_t> levels3720 = {256};
const std::vector<std::vector<size_t>> constShapes3720 = {{1}};
const std::vector<float> fqArgs3720 = {0, 0.631348, 0, 0.631348};
const std::vector<ov::element::Type> modelTypes3720 = {ov::element::f16};
const std::vector<std::vector<ov::Shape>> inputShapes37204d = {{{1, 3, 4, 32}}, {{1, 4, 1, 32}}, {{1, 1, 4, 32}}};
const std::vector<std::vector<ov::Shape>> inputShapes3720nd = {{{3, 8, 128}}, {{4, 8, 128}}};

const auto params3720 =
        ::testing::Combine(::testing::ValuesIn(levels3720), ::testing::ValuesIn(constShapes3720),
                           ::testing::Values(fqArgs3720), ::testing::Values(ov::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(
        smoke_FakeQuantize_ND, FakeQuantizeLayerTest_HW_NPU3720,
        ::testing::Combine(params3720, ::testing::ValuesIn(modelTypes3720),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes3720nd)),
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        FakeQuantizeLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_FakeQuantize_4D, FakeQuantizeLayerTest_HW_NPU3720,
        ::testing::Combine(params3720, ::testing::ValuesIn(modelTypes3720),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes37204d)),
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        FakeQuantizeLayerTest_HW_NPU3720::getTestCaseName);
}  // namespace
