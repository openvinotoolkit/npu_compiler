//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/topk.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace ov {
namespace test {

// Suppression for gtest framework internal test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TopKLayerTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(TopK11LayerTest);

class TopKLayerTestCommon : virtual public TopKLayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "disabled-passes=convert-precision-to-fp";
    }
};
class TopK11LayerTestCommon : public TopK11LayerTest, virtual public VpuOv2LayerTest {};
class TopKDDRLayerTestCommon : public TopK11LayerTest, virtual public VpuOv2LayerTest {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "disabled-passes=convert-precision-to-fp";
    }
};
class TopkDmaLayerTestCommon : public TopKDDRLayerTestCommon {};

constexpr size_t K8_KEY_PATTERNS = 6;

// Payloads for the K=8/axis-0 fast path, which sorts packed keys
// (monotoneU16(value) << 16 | 127 - index) instead of comparing fp16 directly.
// The default filler emits ten distinct non-negative values and never reaches
// that encoding, so these patterns drive the sign-flip half of the mapping, the
// -0.0/+0.0 collapse, the fp16 extremes (-inf and +/-65504, while NaN and +inf
// are left out because they make the reference check ill-defined) and the index
// tie-break. Fixed per-line patterns also turn a stale line pointer into a hard
// mismatch.
void fillK8KeyPatternLine(ov::float16* line, size_t w, size_t patternIdx) {
    const auto set = [line](size_t i, float v) {
        line[i] = ov::float16(v);
    };

    switch (patternIdx) {
    case 0:  // all negative, all distinct: the whole top-8 comes from the flipped half
        for (size_t i = 0; i < w; i++) {
            set(i, -static_cast<float>(i + 1));
        }
        break;
    case 1:  // alternating signs: top-8 straddles the sign boundary
        for (size_t i = 0; i < w; i++) {
            set(i, ((i % 2) ? 1.0f : -1.0f) * static_cast<float>(i + 1));
        }
        break;
    case 2:  // -0.0 and +0.0 are the largest values and must tie by ascending index
        for (size_t i = 0; i < w; i++) {
            set(i, -static_cast<float>(i + 1));
        }
        for (size_t z = 0; z * 11 < w; z++) {
            set(z * 11, (z % 2) ? -0.0f : 0.0f);
        }
        break;
    case 3:  // every element equal: top-8 must be indices 0..7
        for (size_t i = 0; i < w; i++) {
            set(i, 1.0f);
        }
        break;
    case 4:  // fp16 extremes: -inf must sink, the two maxima must tie by index
        for (size_t i = 0; i < w; i++) {
            set(i, static_cast<float>(i) - static_cast<float>(w / 2));
        }
        set(5, -std::numeric_limits<float>::infinity());
        set(w - 28, -std::numeric_limits<float>::infinity());
        set(9, -65504.0f);
        set(3, 65504.0f);
        set(70 % w, 65504.0f);
        break;
    default:  // seven distinct values, then a tie spanning the rank-8 boundary
        for (size_t i = 0; i < w; i++) {
            set(i, 5.0f);
        }
        for (size_t i = 0; i < 7; i++) {
            set(i, 100.0f - static_cast<float>(i));
        }
        break;
    }
}

class TopK_K8Axis0KeyPatternLayerTest : public TopKLayerTestCommon {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        const auto& shape = targetInputStaticShapes[0];
        ASSERT_EQ(funcInputs[0].get_element_type(), ov::element::f16);

        ov::Tensor tensor(funcInputs[0].get_element_type(), shape);
        const size_t w = shape.back();
        const size_t numLines = ov::shape_size(shape) / w;
        auto* data = tensor.data<ov::float16>();
        for (size_t line = 0; line < numLines; line++) {
            fillK8KeyPatternLine(data + line * w, w, line % K8_KEY_PATTERNS);
        }

        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});
    }
};

class TopK_SCFTilingLayerTest : public TopKLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "disabled-passes=convert-precision-to-fp scf-tiling=true";
        // E-190336 for MC support
        VpuOv2LayerTest::configuration["NPU_TILES"] = "1";
    }
};
TEST_P(TopKLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopKLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopK_K8Axis0KeyPatternLayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopK_K8Axis0KeyPatternLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopK_K8Axis0KeyPatternLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(TopK_K8Axis0KeyPatternLayerTest, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

TEST_P(TopK_SCFTilingLayerTest, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopKDDRLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopKDDRLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(TopkDmaLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(TopK_SCFTilingLayerTest, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(TopK11LayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopK11LayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(TopKLayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(TopK11LayerTestCommon, NPU5010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU5010);
}

TEST_P(TopKLayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

TEST_P(TopK11LayerTestCommon, NPU5020_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU5020);
}

TEST_P(TopKLayerTestCommon, NPU6010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU6010);
}

TEST_P(TopK11LayerTestCommon, NPU6010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU6010);
}
TEST_P(TopkDmaLayerTestCommon, NPU6010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}
TEST_P(TopK_SCFTilingLayerTest, NPU6010_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU6010);
}

TEST_P(TopK_K8Axis0KeyPatternLayerTest, NPU6010_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU6010);
}

class TopK1LayerTest : public TopKLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<InputShape> inputShape;
        ov::element::Type modelType;
        int64_t keepK, axis;
        ov::op::v3::TopK::Mode mode;
        ov::op::v3::TopK::SortType sort;
        std::tie(keepK, axis, mode, sort, modelType, inputShape, targetDevice) = this->GetParam();
        init_input_shapes(inputShape);

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front());
        auto k = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, &keepK);
        auto topk = std::dynamic_pointer_cast<ov::op::v3::TopK>(
                std::make_shared<ov::op::v3::TopK>(param, k, axis, mode, sort));

        ov::ResultVector results;
        for (size_t i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "TopK");
    }
};

TEST_P(TopK1LayerTest, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace test
}  // namespace ov

using ov::test::TopK11LayerTestCommon;
using ov::test::TopK1LayerTest;
using ov::test::TopK_K8Axis0KeyPatternLayerTest;
using ov::test::TopK_SCFTilingLayerTest;
using ov::test::TopKDDRLayerTestCommon;
using ov::test::TopkDmaLayerTestCommon;
using ov::test::TopKLayerTestCommon;

namespace {

const std::vector<ov::element::Type> modelTypeFP16 = {ov::element::f16};
// SI32 data type is currently unsupported by OpenVINO TopK test environment
const std::vector<ov::element::Type> modelTypes = {ov::element::f32 /*, ov::element::i32*/};

const std::vector<int64_t> axes = {0, 1, 2};

const std::vector<int64_t> k = {1, 5, 10};

const std::vector<ov::op::v3::TopK::Mode> modes = {ov::op::v3::TopK::Mode::MIN, ov::op::v3::TopK::Mode::MAX};

const std::vector<ov::op::v3::TopK::SortType> sortTypes = {
        // The implements of SortType::NONE are different.
        // Reference uses std::nth_element and returns k out-of-order values.
        // Kernel returns k data sorted in values. nth_element causes computation increase.
        // ov::op::v3::TopK::SortType::NONE,
        ov::op::v3::TopK::SortType::SORT_INDICES,
        ov::op::v3::TopK::SortType::SORT_VALUES,
};

const auto paramsConfig = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{1, 5}), ::testing::ValuesIn(axes), ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(modelTypeFP16),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{5, 5, 5}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK, TopKLayerTestCommon, paramsConfig, TopKLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK1, TopK1LayerTest, paramsConfig, TopK1LayerTest::getTestCaseName);

const auto paramsConfigPrecommitFP32 = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{1}), ::testing::ValuesIn(std::vector<int64_t>{2}),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{5, 5, 5}}}))),
        ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK_FP32, TopKLayerTestCommon, paramsConfigPrecommitFP32,
                         TopKLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_FP32_SCFTiling, TopK_SCFTilingLayerTest, paramsConfigPrecommitFP32,
                         TopK_SCFTilingLayerTest::getTestCaseName);

// Tiling tests
const std::vector<int64_t> k_Tilling = {1};
const std::vector<int64_t> axes_Tilling = {1};
const std::vector<ov::op::v3::TopK::Mode> modes_Tilling = {ov::op::v3::TopK::Mode::MAX};
const std::vector<ov::op::v3::TopK::SortType> sortTypes_Tilling = {
        ov::op::v3::TopK::SortType::SORT_INDICES,
};
const std::vector<ov::element::Type> modelTypes_Tilling = {ov::element::f16};
const std::vector<std::vector<ov::Shape>> inShapes = {{{1, 8, 16, 21}}, {{1, 8, 16, 32}}};
const std::vector<std::vector<ov::Shape>> inShapes_opset11 = {{{1, 300, 8}}, {{1, 151, 7049}}};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_Tilling, TopKLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling), ::testing::ValuesIn(axes_Tilling),
                                            ::testing::ValuesIn(modes_Tilling), ::testing::ValuesIn(sortTypes_Tilling),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 5, 512, 512}}}))),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKLayerTestCommon::getTestCaseName);

// K=1 asm optimization tests
INSTANTIATE_TEST_SUITE_P(smoke_TopK_K1, TopKLayerTestCommon,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling),
                                            ::testing::ValuesIn(std::vector<int64_t>{2}),
                                            ::testing::ValuesIn(modes_Tilling),
                                            ::testing::ValuesIn(std::vector<ov::op::v3::TopK::SortType>{
                                                    ov::op::v3::TopK::SortType::NONE}),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>{{{1, 42840, 13}}})),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_TopK_AxisValUpTo32, TopKLayerTestCommon,
        ::testing::Combine(::testing::ValuesIn(std::vector<int64_t>{1}), ::testing::ValuesIn(std::vector<int64_t>{3}),
                           ::testing::ValuesIn(modes_Tilling),
                           ::testing::ValuesIn(std::vector<ov::op::v3::TopK::SortType>{
                                   ov::op::v3::TopK::SortType::SORT_VALUES}),
                           ::testing::ValuesIn(modelTypes_Tilling),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                           ::testing::Values(test_utils::TARGET_DEVICE)),
        TopKLayerTestCommon::getTestCaseName);

// Reduced (last) axis lengths chosen to exercise every branch of the K=8/axis-0
// fold: the initial K-element seed, the scalar bridge that aligns j from K to the
// first VW=32 boundary, the 32-lane SIMD chunk loop with its chunk-skip gate, and
// the final scalar tail loop. Chunk count for W >= VW is floor((W - VW) / VW).
// Stable (value desc, index asc) tie-break is covered by the default filler
// (InputGenerateData{0,10,1} => 10 distinct fp16 values), which guarantees
// duplicates for any N>=11 and a tied rank-8 boundary with probability >99.95%
// for N>=96.
const std::vector<std::vector<ov::Shape>> inShapes_K8Axis0 = {
        {{64, 8}},       // W == k: seed-only path, no bridge/chunk/tail
        {{64, 32}},      // W == VW: bridge only, no chunk, no tail
        {{64, 33}},      // bridge + 1-element tail, no chunk
        {{64, 96}},      // bridge + 2 full chunks, no tail
        {{64, 100}},     // bridge + 2 full chunks + 4-element tail
        {{64, 128}},     // Qwen3-30B-A3B router width: bridge + 3 full chunks
        {{8, 16, 257}},  // bridge + 7 full chunks + 1-element tail, rank-3
        {{64, 4096}},    // bridge + 127 full chunks, wide line
        {{4, 8192}},     // maximum supported reduced-axis length: bridge + 255 chunks
};

const auto paramsK8Axis0 =
        ::testing::Combine(::testing::Values(int64_t{8}),   // k == 8
                           ::testing::Values(int64_t{-1}),  // reduce last (contiguous) axis -> kernel axis 0
                           ::testing::Values(ov::op::v3::TopK::Mode::MAX),              // mode == MAX
                           ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),  // sort == SORT_VALUES
                           ::testing::ValuesIn(modelTypeFP16),                          // fp16
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes_K8Axis0)),
                           ::testing::Values(test_utils::TARGET_DEVICE));

INSTANTIATE_TEST_SUITE_P(smoke_TopK_K8Axis0, TopKLayerTestCommon, paramsK8Axis0, TopKLayerTestCommon::getTestCaseName);

// Same fast-path configuration, but with payloads that reach the key encoding:
// negatives, signed zeros, fp16 extremes and ties across several 128-wide lines.
const std::vector<std::vector<ov::Shape>> inShapes_K8Axis0KeyPattern = {
        {{6, 128}},     // one line per pattern
        {{12, 128}},    // every pattern twice: a stale line pointer repeats a line
        {{2, 3, 128}},  // same patterns through the rank-3 dense path
};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_K8Axis0KeyPattern, TopK_K8Axis0KeyPatternLayerTest,
                         ::testing::Combine(::testing::Values(int64_t{8}), ::testing::Values(int64_t{-1}),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::ValuesIn(modelTypeFP16),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    inShapes_K8Axis0KeyPattern)),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopK_K8Axis0KeyPatternLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_SCFTiling, TopK_SCFTilingLayerTest,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling), ::testing::ValuesIn(axes_Tilling),
                                            ::testing::ValuesIn(modes_Tilling), ::testing::ValuesIn(sortTypes_Tilling),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 5, 512, 512}}}))),
                                            ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopK_SCFTilingLayerTest::getTestCaseName);

}  // namespace

namespace {  // opset v11

INSTANTIATE_TEST_SUITE_P(smoke_TopK11, TopK11LayerTestCommon,
                         ::testing::Combine(::testing::Values(1), ::testing::Values(1),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_INDICES),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{10, 10, 10}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopK11LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK11_GPT, TopK11LayerTestCommon,
                         ::testing::Combine(::testing::Values(4), ::testing::Values(3),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{1, 1, 1024, 32}}}))),
                                            ::testing::Values(false), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopK11LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_TopK11_K1, TopK11LayerTestCommon,
        ::testing::Combine(::testing::Values(1), ::testing::Values(2), ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                           ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                           ::testing::Values(ov::element::f16),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes_opset11)),
                           ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
        TopK11LayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK11_K300, TopK11LayerTestCommon,
                         ::testing::Combine(::testing::Values(300), ::testing::Values(1),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{1, 3600}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopK11LayerTestCommon::getTestCaseName);

// Can't tile, require DDR
INSTANTIATE_TEST_SUITE_P(smoke_TopK11_conformance, TopKDDRLayerTestCommon,
                         ::testing::Combine(::testing::Values(1), ::testing::Values(3),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{1, 513, 513, 21}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKDDRLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK11_DDRAccess, TopKDDRLayerTestCommon,
                         ::testing::Combine(::testing::Values(1), ::testing::Values(-1),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{1, 5898240}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopKDDRLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK11_Dma, TopkDmaLayerTestCommon,
                         ::testing::Combine(::testing::Values(1, 100, 300), ::testing::Values(-1),
                                            ::testing::Values(ov::op::v3::TopK::Mode::MAX, ov::op::v3::TopK::Mode::MIN),
                                            ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES),
                                            ::testing::Values(ov::element::f16, ov::element::i32),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(
                                                    std::vector<ov::Shape>({{{1, 1, 17, 399999}}}))),
                                            ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
                         TopkDmaLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_TopK11_DDRAccess_LargeLineBuffer, TopKDDRLayerTestCommon,
        ::testing::Combine(
                ::testing::Values(1), ::testing::Values(0), ::testing::Values(ov::op::v3::TopK::Mode::MAX),
                ::testing::Values(ov::op::v3::TopK::SortType::SORT_VALUES), ::testing::Values(ov::element::f32),
                ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{250112}}))),
                ::testing::Values(true), ::testing::Values(test_utils::TARGET_DEVICE)),
        TopKDDRLayerTestCommon::getTestCaseName);

}  // namespace
