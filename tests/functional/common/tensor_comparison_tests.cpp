//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/npu_test_env_cfg.hpp"
#include "common/tensor_comparison.hpp"

#include <gtest/gtest.h>
#include <openvino/core/type/float16.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

using namespace ov::test::utils;

using TensorComparisonTestParams = std::string;

class TensorComparison : public testing::TestWithParam<TensorComparisonTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TensorComparisonTestParams>& obj) {
        std::string targetDevice = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "target_platform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "target_device=" << targetDevice << "_";
        return result.str();
    }
};

namespace {

ov::Tensor makeF32Tensor(const std::vector<float>& values) {
    ov::Tensor tensor(ov::element::f32, {values.size()});
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    return tensor;
}

ov::Tensor makeF16Tensor(const std::vector<ov::float16>& values) {
    ov::Tensor tensor(ov::element::f16, {values.size()});
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }
    return tensor;
}

const auto INF = std::numeric_limits<float>::infinity();
const auto NAN_F = std::numeric_limits<float>::quiet_NaN();
const auto F16_MAX = std::numeric_limits<ov::float16>::max();
const auto F16_LOWEST = std::numeric_limits<ov::float16>::lowest();
const auto F16_INF = std::numeric_limits<ov::float16>::infinity();
const auto F16_NEG_INF = ov::float16::from_bits(0xFC00);

// --- Normal operation ---

TEST_P(TensorComparison, ExactMatch) {
    auto expected = makeF32Tensor({1.0f, 2.0f, 3.0f});
    auto actual = makeF32Tensor({1.0f, 2.0f, 3.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_EQ(result.mismatchCount, 0u);
    EXPECT_EQ(result.totalElements, 3u);
    EXPECT_DOUBLE_EQ(result.expectedMin, 1.0);
    EXPECT_DOUBLE_EQ(result.expectedMax, 3.0);
    EXPECT_DOUBLE_EQ(result.actualMin, 1.0);
    EXPECT_DOUBLE_EQ(result.actualMax, 3.0);
}

TEST_P(TensorComparison, WithinAbsThreshold) {
    auto expected = makeF32Tensor({1.0f, 2.0f});
    auto actual = makeF32Tensor({1.05f, 2.05f});
    auto result = compareTensors(expected, actual, 0.1, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_EQ(result.mismatchCount, 0u);
}

TEST_P(TensorComparison, WithinRelThreshold) {
    auto expected = makeF32Tensor({100.0f, 200.0f});
    auto actual = makeF32Tensor({101.0f, 202.0f});
    // tolerance = 0 + 0.02 * |expected| = 2.0 and 4.0 respectively
    auto result = compareTensors(expected, actual, 0.0, 0.02);

    EXPECT_TRUE(result.passed());
    EXPECT_EQ(result.mismatchCount, 0u);
}

TEST_P(TensorComparison, AbsThresholdExceeded) {
    auto expected = makeF32Tensor({1.0f, 2.0f, 3.0f});
    auto actual = makeF32Tensor({1.0f, 2.5f, 3.0f});
    auto result = compareTensors(expected, actual, 0.1, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 1u);
}

TEST_P(TensorComparison, RelThresholdExceeded) {
    auto expected = makeF32Tensor({100.0f});
    auto actual = makeF32Tensor({120.0f});
    // tolerance = 0 + 0.01 * 100 = 1.0, diff = 20.0
    auto result = compareTensors(expected, actual, 0.0, 0.01);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 1u);
}

TEST_P(TensorComparison, MixedPassFail) {
    auto expected = makeF32Tensor({1.0f, 2.0f, 3.0f, 4.0f});
    auto actual = makeF32Tensor({1.0f, 5.0f, 3.0f, 7.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 2u);
    EXPECT_EQ(result.totalElements, 4u);
    EXPECT_DOUBLE_EQ(result.mismatchPercentage(), 50.0);
}

TEST_P(TensorComparison, WorstMismatchesSorted) {
    auto expected = makeF32Tensor({0.0f, 0.0f, 0.0f, 0.0f});
    auto actual = makeF32Tensor({1.0f, 3.0f, 2.0f, 0.5f});
    auto result = compareTensors(expected, actual, 0.0, 0.0, /*topN=*/10);

    EXPECT_EQ(result.worstMismatches.size(), 4u);
    for (size_t i = 1; i < result.worstMismatches.size(); ++i) {
        EXPECT_GE(result.worstMismatches[i - 1].absError, result.worstMismatches[i].absError)
                << "worstMismatches not sorted descending at index " << i;
    }
    EXPECT_DOUBLE_EQ(result.worstMismatches[0].absError, 3.0);
}

TEST_P(TensorComparison, TopNLimitsResults) {
    auto expected = makeF32Tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    auto actual = makeF32Tensor({1.0f, 5.0f, 3.0f, 2.0f, 4.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0, /*topN=*/3);

    EXPECT_EQ(result.mismatchCount, 5u);
    EXPECT_EQ(result.worstMismatches.size(), 3u);
    // Should keep the 3 largest errors: 5.0, 4.0, 3.0
    EXPECT_DOUBLE_EQ(result.worstMismatches[0].absError, 5.0);
    EXPECT_DOUBLE_EQ(result.worstMismatches[1].absError, 4.0);
    EXPECT_DOUBLE_EQ(result.worstMismatches[2].absError, 3.0);
}

TEST_P(TensorComparison, RangesCorrect) {
    auto expected = makeF32Tensor({-5.0f, 0.0f, 10.0f});
    auto actual = makeF32Tensor({-3.0f, 1.0f, 8.0f});
    auto result = compareTensors(expected, actual, 100.0, 0.0);

    EXPECT_DOUBLE_EQ(result.expectedMin, -5.0);
    EXPECT_DOUBLE_EQ(result.expectedMax, 10.0);
    EXPECT_DOUBLE_EQ(result.actualMin, -3.0);
    EXPECT_DOUBLE_EQ(result.actualMax, 8.0);
}

// --- Edge cases ---

TEST_P(TensorComparison, AllNaN) {
    auto expected = makeF32Tensor({NAN_F, NAN_F, NAN_F});
    auto actual = makeF32Tensor({NAN_F, NAN_F, NAN_F});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_EQ(result.expectedNanCount, 3u);
    EXPECT_EQ(result.actualNanCount, 3u);

    auto formatted = formatComparisonResult(result, "all_nan");
    EXPECT_NE(formatted.find("N/A (all NaN)"), std::string::npos) << "Expected 'N/A (all NaN)' in output, got:\n"
                                                                  << formatted;
}

TEST_P(TensorComparison, TopNZero) {
    auto expected = makeF32Tensor({1.0f, 2.0f});
    auto actual = makeF32Tensor({9.0f, 9.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0, /*topN=*/0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 2u);
    EXPECT_TRUE(result.worstMismatches.empty());
}

TEST_P(TensorComparison, HeapDrainAll) {
    auto expected = makeF32Tensor({0.0f, 0.0f, 0.0f});
    auto actual = makeF32Tensor({1.0f, 2.0f, 3.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0, /*topN=*/10);

    EXPECT_EQ(result.mismatchCount, 3u);
    EXPECT_EQ(result.worstMismatches.size(), 3u);
    // Verify all entries have correct values (sorted descending)
    EXPECT_DOUBLE_EQ(result.worstMismatches[0].absError, 3.0);
    EXPECT_DOUBLE_EQ(result.worstMismatches[1].absError, 2.0);
    EXPECT_DOUBLE_EQ(result.worstMismatches[2].absError, 1.0);
}

TEST_P(TensorComparison, ShapeMismatch) {
    ov::Tensor expected(ov::element::f32, {2, 3});
    ov::Tensor actual(ov::element::f32, {3, 2});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_FALSE(result.errorMessage.empty());
    EXPECT_NE(result.errorMessage.find("Shape mismatch"), std::string::npos);
}

TEST_P(TensorComparison, ElementTypeMismatch) {
    ov::Tensor expected(ov::element::f32, {4});
    ov::Tensor actual(ov::element::f16, {4});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_FALSE(result.errorMessage.empty());
    EXPECT_NE(result.errorMessage.find("Element type mismatch"), std::string::npos);
}

TEST_P(TensorComparison, InfMatchSameSign) {
    auto expected = makeF32Tensor({INF, -INF, 1.0f});
    auto actual = makeF32Tensor({INF, -INF, 1.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_EQ(result.mismatchCount, 0u);
}

TEST_P(TensorComparison, InfMismatchDifferentSign) {
    auto expected = makeF32Tensor({INF});
    auto actual = makeF32Tensor({-INF});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 1u);
}

TEST_P(TensorComparison, InfVsFinite) {
    auto expected = makeF32Tensor({INF});
    auto actual = makeF32Tensor({0.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 1u);
}

TEST_P(TensorComparison, OneNaNMismatch) {
    auto expected = makeF32Tensor({NAN_F, 1.0f});
    auto actual = makeF32Tensor({1.0f, NAN_F});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 2u);
    EXPECT_EQ(result.expectedNanCount, 1u);
    EXPECT_EQ(result.actualNanCount, 1u);
}

TEST_P(TensorComparison, BothNaNMatch) {
    auto expected = makeF32Tensor({NAN_F, 1.0f, 2.0f});
    auto actual = makeF32Tensor({NAN_F, 1.0f, 2.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_EQ(result.mismatchCount, 0u);
    EXPECT_EQ(result.expectedNanCount, 1u);
    EXPECT_EQ(result.actualNanCount, 1u);
}

// --- Overflow at type boundary (fp16) ---

TEST_P(TensorComparison, OverflowAtUpperBound_F16) {
    // expected = fp16 max (65504), actual = +Inf: should be treated as match
    // (hardware overflow at the type's representable limit)
    auto expected = makeF16Tensor({ov::float16(1.0f), F16_MAX});
    auto actual = makeF16Tensor({ov::float16(1.0f), F16_INF});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed()) << formatComparisonResult(result, "OverflowAtUpperBound_F16");
    EXPECT_EQ(result.mismatchCount, 0u);
}

TEST_P(TensorComparison, OverflowAtUpperBound_Reversed_F16) {
    // expected = +Inf, actual = fp16 max: same case, reversed
    auto expected = makeF16Tensor({F16_INF});
    auto actual = makeF16Tensor({F16_MAX});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed()) << formatComparisonResult(result, "OverflowAtUpperBound_Reversed_F16");
    EXPECT_EQ(result.mismatchCount, 0u);
}

TEST_P(TensorComparison, OverflowAtLowerBound_F16) {
    // expected = fp16 lowest (-65504), actual = -Inf: should be treated as match
    auto expected = makeF16Tensor({F16_LOWEST});
    auto actual = makeF16Tensor({F16_NEG_INF});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed()) << formatComparisonResult(result, "OverflowAtLowerBound_F16");
    EXPECT_EQ(result.mismatchCount, 0u);
}

TEST_P(TensorComparison, InfVsFiniteNotAtBound_F16) {
    // expected = 100.0 (not at fp16 max), actual = Inf: should be a mismatch
    auto expected = makeF16Tensor({ov::float16(100.0f)});
    auto actual = makeF16Tensor({F16_INF});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 1u);
}

TEST_P(TensorComparison, DifferentSignOverflow_F16) {
    // expected = fp16 max (+65504), actual = -Inf: different sign, should be a mismatch
    auto expected = makeF16Tensor({F16_MAX});
    auto actual = makeF16Tensor({F16_NEG_INF});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatchCount, 1u);
}

TEST_P(TensorComparison, OverflowAtUpperBound_F32) {
    // Same logic works for f32: expected = f32 max, actual = +Inf
    const auto F32_MAX = std::numeric_limits<float>::max();
    auto expected = makeF32Tensor({F32_MAX});
    auto actual = makeF32Tensor({INF});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed()) << formatComparisonResult(result, "OverflowAtUpperBound_F32");
    EXPECT_EQ(result.mismatchCount, 0u);
}

// --- Expected anomaly warnings ---

TEST_P(TensorComparison, NoExpectedAnomalies) {
    auto expected = makeF32Tensor({1.0f, 2.0f, 3.0f});
    auto actual = makeF32Tensor({1.0f, 2.0f, 3.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_FALSE(result.hasExpectedAnomalies());
    EXPECT_TRUE(formatExpectedAnomalyWarning(result).empty());
}

TEST_P(TensorComparison, ExpectedNanAnomaly) {
    auto expected = makeF32Tensor({NAN_F, 1.0f});
    auto actual = makeF32Tensor({NAN_F, 1.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_TRUE(result.hasExpectedAnomalies());
    auto warning = formatExpectedAnomalyWarning(result, "test");
    EXPECT_NE(warning.find("NaN"), std::string::npos);
    EXPECT_NE(warning.find("WARNING"), std::string::npos);
}

TEST_P(TensorComparison, ExpectedInfAnomaly) {
    auto expected = makeF32Tensor({INF, 1.0f});
    auto actual = makeF32Tensor({INF, 1.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.passed());
    EXPECT_TRUE(result.hasExpectedAnomalies());
    auto warning = formatExpectedAnomalyWarning(result, "test");
    EXPECT_NE(warning.find("Inf"), std::string::npos);
}

TEST_P(TensorComparison, ExpectedNanAndInfAnomaly) {
    auto expected = makeF32Tensor({NAN_F, INF, 1.0f});
    auto actual = makeF32Tensor({NAN_F, INF, 1.0f});
    auto result = compareTensors(expected, actual, 0.0, 0.0);

    EXPECT_TRUE(result.hasExpectedAnomalies());
    auto warning = formatExpectedAnomalyWarning(result);
    EXPECT_NE(warning.find("NaN"), std::string::npos);
    EXPECT_NE(warning.find("Inf"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(smoke_LayerTest_TensorComparison, TensorComparison,
                         ::testing::Values(test_utils::TARGET_DEVICE), TensorComparison::getTestCaseName);

}  // namespace
