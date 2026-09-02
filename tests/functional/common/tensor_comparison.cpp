//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/tensor_comparison.hpp"

#include <gtest/gtest.h>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/core/type/float4_e2m1.hpp>
#include <openvino/core/type/float8_e4m3.hpp>
#include <openvino/core/type/float8_e5m2.hpp>
#include <openvino/core/type/float8_e8m0.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>

namespace ov::test::utils {

namespace {

std::vector<size_t> flatToMultiIndex(size_t flatIndex, const ov::Shape& shape) {
    std::vector<size_t> result(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        result[i] = flatIndex % shape[i];
        flatIndex /= shape[i];
    }
    return result;
}

std::string formatIndex(const std::vector<size_t>& idx) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < idx.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << idx[i];
    }
    ss << "]";
    return ss.str();
}

// Min-heap entry: we keep topN largest errors, so we use a min-heap and evict the smallest
struct HeapEntry {
    double absError;
    size_t flatIndex;
    double expected;
    double actual;
    double tolerance;

    bool operator>(const HeapEntry& other) const {
        return absError > other.absError;
    }
};

template <typename T>
TensorComparisonResult compareTensorsTyped(const T* expectedData, const T* actualData, const ov::Shape& shape,
                                           ov::element::Type elementType, double absThreshold, double relThreshold,
                                           size_t topN) {
    TensorComparisonResult result;
    result.shape = shape;
    result.elementType = elementType;
    result.totalElements = ov::shape_size(shape);
    result.absThreshold = absThreshold;
    result.relThreshold = relThreshold;

    // Sentinel value -1 means "not set" — treat as zero (disabled)
    const auto effectiveAbsThreshold = absThreshold < 0 ? 0.0 : absThreshold;
    const auto effectiveRelThreshold = relThreshold < 0 ? 0.0 : relThreshold;

    result.expectedMin = std::numeric_limits<double>::max();
    result.expectedMax = std::numeric_limits<double>::lowest();
    result.actualMin = std::numeric_limits<double>::max();
    result.actualMax = std::numeric_limits<double>::lowest();

    // Min-heap of top N worst mismatches (smallest absError at top, so we can evict it)
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;

    for (size_t i = 0; i < result.totalElements; ++i) {
        const auto e = static_cast<double>(expectedData[i]);
        const auto a = static_cast<double>(actualData[i]);

        // NaN/Inf tracking
        if (std::isnan(e)) {
            ++result.expectedNanCount;
        }
        if (std::isinf(e)) {
            ++result.expectedInfCount;
        }
        if (std::isnan(a)) {
            ++result.actualNanCount;
        }
        if (std::isinf(a)) {
            ++result.actualInfCount;
        }

        // Both NaN = match
        if (std::isnan(e) && std::isnan(a)) {
            continue;
        }

        // Range tracking (skip NaN)
        if (!std::isnan(e)) {
            result.expectedMin = std::min(result.expectedMin, e);
            result.expectedMax = std::max(result.expectedMax, e);
        }
        if (!std::isnan(a)) {
            result.actualMin = std::min(result.actualMin, a);
            result.actualMax = std::max(result.actualMax, a);
        }

        // One NaN, one not = mismatch
        if (std::isnan(e) || std::isnan(a)) {
            ++result.mismatchCount;
            const double inf = std::numeric_limits<double>::infinity();
            if (topN > 0) {
                HeapEntry entry{inf, i, e, a, effectiveAbsThreshold + effectiveRelThreshold * std::abs(e)};
                if (heap.size() < topN) {
                    heap.push(entry);
                } else if (entry.absError > heap.top().absError) {
                    heap.pop();
                    heap.push(entry);
                }
            }
            continue;
        }

        // Infinity handling: match only when both are inf with the same sign
        if (std::isinf(e) || std::isinf(a)) {
            if (e == a) {
                continue;
            }

            // Skip overflow at type boundary: one value at the type's representable limit,
            // the other overflowed to Inf in the same sign direction.
            // This is expected for low-precision types (e.g. fp16) where hardware arithmetic
            // may overflow slightly before or after the reference implementation.
            if constexpr (std::numeric_limits<T>::is_specialized && !std::numeric_limits<T>::is_integer) {
                const auto typeMax = static_cast<double>(std::numeric_limits<T>::max());
                const auto typeLowest = static_cast<double>(std::numeric_limits<T>::lowest());

                const bool overflowAtUpperBound =
                        (e >= typeMax && std::isinf(a) && a > 0) || (a >= typeMax && std::isinf(e) && e > 0);
                const bool overflowAtLowerBound =
                        (e <= typeLowest && std::isinf(a) && a < 0) || (a <= typeLowest && std::isinf(e) && e < 0);

                if (overflowAtUpperBound || overflowAtLowerBound) {
                    continue;
                }
            }

            ++result.mismatchCount;
            const double inf = std::numeric_limits<double>::infinity();
            if (topN > 0) {
                HeapEntry entry{inf, i, e, a, effectiveAbsThreshold + effectiveRelThreshold * std::abs(e)};
                if (heap.size() < topN) {
                    heap.push(entry);
                } else if (entry.absError > heap.top().absError) {
                    heap.pop();
                    heap.push(entry);
                }
            }
            continue;
        }

        const auto absDiff = std::abs(e - a);
        const auto tolerance = effectiveAbsThreshold + effectiveRelThreshold * std::abs(e);

        if (absDiff > tolerance) {
            ++result.mismatchCount;
            if (topN > 0) {
                HeapEntry entry{absDiff, i, e, a, tolerance};
                if (heap.size() < topN) {
                    heap.push(entry);
                } else if (absDiff > heap.top().absError) {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }
    }

    // Extract heap into sorted vector (descending by absError)
    const auto heapSize = static_cast<int>(heap.size());
    result.worstMismatches.resize(heapSize);
    for (int i = heapSize - 1; i >= 0; --i) {
        const auto& top = heap.top();
        result.worstMismatches[i] = MismatchInfo{top.flatIndex, flatToMultiIndex(top.flatIndex, shape),
                                                 top.expected,  top.actual,
                                                 top.absError,  top.tolerance};
        heap.pop();
    }

    return result;
}

// Sub-byte 4-bit types are packed 2 elements per byte:
// element[idx] is at byte[idx/2], bits 3:0 if idx is even, bits 7:4 if idx is odd.
TensorComparisonResult compareU4Tensors(const ov::Tensor& expected, const ov::Tensor& actual, const ov::Shape& shape,
                                        double absThreshold, double relThreshold, size_t topN) {
    const size_t nElem = ov::shape_size(shape);
    const auto* expBytes = static_cast<const uint8_t*>(expected.data());
    const auto* actBytes = static_cast<const uint8_t*>(actual.data());

    std::vector<uint8_t> expUnpacked(nElem), actUnpacked(nElem);
    for (size_t i = 0; i < nElem; ++i) {
        expUnpacked[i] = (expBytes[i / 2] >> (4 * (i % 2))) & 0xF;
        actUnpacked[i] = (actBytes[i / 2] >> (4 * (i % 2))) & 0xF;
    }
    return compareTensorsTyped(expUnpacked.data(), actUnpacked.data(), shape, ov::element::u4, absThreshold,
                               relThreshold, topN);
}

TensorComparisonResult compareI4Tensors(const ov::Tensor& expected, const ov::Tensor& actual, const ov::Shape& shape,
                                        double absThreshold, double relThreshold, size_t topN) {
    const size_t nElem = ov::shape_size(shape);
    const auto* expBytes = static_cast<const uint8_t*>(expected.data());
    const auto* actBytes = static_cast<const uint8_t*>(actual.data());

    std::vector<int8_t> expUnpacked(nElem), actUnpacked(nElem);
    for (size_t i = 0; i < nElem; ++i) {
        const uint8_t expNibble = (expBytes[i / 2] >> (4 * (i % 2))) & 0xF;
        const uint8_t actNibble = (actBytes[i / 2] >> (4 * (i % 2))) & 0xF;
        expUnpacked[i] = static_cast<int8_t>(expNibble | (expNibble & 0x8 ? 0xF0 : 0));
        actUnpacked[i] = static_cast<int8_t>(actNibble | (actNibble & 0x8 ? 0xF0 : 0));
    }
    return compareTensorsTyped(expUnpacked.data(), actUnpacked.data(), shape, ov::element::i4, absThreshold,
                               relThreshold, topN);
}

}  // namespace

TensorComparisonResult compareTensors(const ov::Tensor& expected, const ov::Tensor& actual, double absThreshold,
                                      double relThreshold, size_t topN) {
    if (expected.get_shape() != actual.get_shape()) {
        std::ostringstream msg;
        msg << "Shape mismatch: expected " << expected.get_shape() << " vs actual " << actual.get_shape();
        TensorComparisonResult fail;
        fail.shape = expected.get_shape();
        fail.elementType = expected.get_element_type();
        fail.totalElements = ov::shape_size(expected.get_shape());
        fail.errorMessage = msg.str();
        return fail;
    }

    if (expected.get_element_type() != actual.get_element_type()) {
        std::ostringstream msg;
        msg << "Element type mismatch: expected " << expected.get_element_type() << " vs actual "
            << actual.get_element_type();
        TensorComparisonResult fail;
        fail.shape = expected.get_shape();
        fail.elementType = expected.get_element_type();
        fail.totalElements = ov::shape_size(expected.get_shape());
        fail.errorMessage = msg.str();
        return fail;
    }
    const auto& shape = expected.get_shape();
    const auto elementType = expected.get_element_type();

    switch (elementType) {
    case ov::element::f32:
        return compareTensorsTyped(expected.data<const float>(), actual.data<const float>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::f64:
        return compareTensorsTyped(expected.data<const double>(), actual.data<const double>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::f16:
        return compareTensorsTyped(expected.data<const ov::float16>(), actual.data<const ov::float16>(), shape,
                                   elementType, absThreshold, relThreshold, topN);
    case ov::element::bf16:
        return compareTensorsTyped(expected.data<const ov::bfloat16>(), actual.data<const ov::bfloat16>(), shape,
                                   elementType, absThreshold, relThreshold, topN);
    case ov::element::f8e4m3:
        return compareTensorsTyped(expected.data<const ov::float8_e4m3>(), actual.data<const ov::float8_e4m3>(), shape,
                                   elementType, absThreshold, relThreshold, topN);
    case ov::element::f8e5m2:
        return compareTensorsTyped(expected.data<const ov::float8_e5m2>(), actual.data<const ov::float8_e5m2>(), shape,
                                   elementType, absThreshold, relThreshold, topN);
    case ov::element::f4e2m1:
        return compareTensorsTyped(expected.data<const ov::float4_e2m1>(), actual.data<const ov::float4_e2m1>(), shape,
                                   elementType, absThreshold, relThreshold, topN);
    case ov::element::f8e8m0:
        return compareTensorsTyped(expected.data<const ov::float8_e8m0>(), actual.data<const ov::float8_e8m0>(), shape,
                                   elementType, absThreshold, relThreshold, topN);
    case ov::element::i64:
        return compareTensorsTyped(expected.data<const int64_t>(), actual.data<const int64_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::i32:
        return compareTensorsTyped(expected.data<const int32_t>(), actual.data<const int32_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::i16:
        return compareTensorsTyped(expected.data<const int16_t>(), actual.data<const int16_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::i8:
        return compareTensorsTyped(expected.data<const int8_t>(), actual.data<const int8_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::u8:
        return compareTensorsTyped(expected.data<const uint8_t>(), actual.data<const uint8_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::u32:
        return compareTensorsTyped(expected.data<const uint32_t>(), actual.data<const uint32_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::boolean: {
        using BooleanTensorValue = ov::fundamental_type_for<ov::element::boolean>;
        return compareTensorsTyped(expected.data<const BooleanTensorValue>(), actual.data<const BooleanTensorValue>(),
                                   shape, elementType, absThreshold, relThreshold, topN);
    }
    case ov::element::u16:
        return compareTensorsTyped(expected.data<const uint16_t>(), actual.data<const uint16_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::u64:
        return compareTensorsTyped(expected.data<const uint64_t>(), actual.data<const uint64_t>(), shape, elementType,
                                   absThreshold, relThreshold, topN);
    case ov::element::u4:
        return compareU4Tensors(expected, actual, shape, absThreshold, relThreshold, topN);
    case ov::element::i4:
        return compareI4Tensors(expected, actual, shape, absThreshold, relThreshold, topN);
    default:
        std::ostringstream msg;
        msg << "Unsupported element type for tensor comparison: " << elementType;
        TensorComparisonResult fail{};
        fail.shape = shape;
        fail.elementType = elementType;
        fail.totalElements = ov::shape_size(shape);
        fail.absThreshold = absThreshold;
        fail.relThreshold = relThreshold;
        fail.errorMessage = msg.str();
        return fail;
    }
}

std::string formatComparisonResult(const TensorComparisonResult& result, std::string_view tensorName) {
    std::ostringstream ss;
    ss << std::fixed;

    // Header
    ss << "Tensor";
    if (!tensorName.empty()) {
        ss << " \"" << tensorName << "\"";
    }
    ss << " " << result.shape << " (" << result.elementType << "):\n";

    if (!result.errorMessage.empty()) {
        ss << "  Error: " << result.errorMessage << "\n";
        return ss.str();
    }

    // Thresholds
    ss << "  absolute: " << std::setprecision(6) << result.absThreshold << ", relative: " << result.relThreshold
       << "  (tolerance = absolute + relative * |expected|)\n";

    // NaN/Inf
    if (result.expectedNanCount > 0 || result.actualNanCount > 0 || result.expectedInfCount > 0 ||
        result.actualInfCount > 0) {
        ss << "  NaN: expected=" << result.expectedNanCount << ", actual=" << result.actualNanCount
           << " | Inf: expected=" << result.expectedInfCount << ", actual=" << result.actualInfCount << "\n";
    }

    // Mismatches
    ss << "  Mismatches: " << result.mismatchCount << " / " << result.totalElements << " (" << std::setprecision(3)
       << result.mismatchPercentage() << "%)\n";

    // Ranges
    if (result.totalElements > 0) {
        ss << std::setprecision(6);
        ss << "  Expected range: ";
        if (result.expectedNanCount < result.totalElements) {
            ss << "[" << result.expectedMin << ", " << result.expectedMax << "]\n";
        } else {
            ss << "N/A (all NaN)\n";
        }
        ss << "  Actual range:   ";
        if (result.actualNanCount < result.totalElements) {
            ss << "[" << result.actualMin << ", " << result.actualMax << "]\n";
        } else {
            ss << "N/A (all NaN)\n";
        }
    }

    // Worst mismatches table
    if (!result.worstMismatches.empty()) {
        // Compute column widths from data
        size_t maxIndexWidth = 5;  // minimum "index"
        for (const auto& m : result.worstMismatches) {
            maxIndexWidth = std::max(maxIndexWidth, formatIndex(m.multiIndex).size());
        }

        ss << "  Worst mismatches:\n";
        ss << "    " << std::left << std::setw(3) << "#" << std::setw(static_cast<int>(maxIndexWidth + 2)) << "index"
           << std::right << std::setw(14) << "expected" << std::setw(14) << "actual" << std::setw(14) << "diff"
           << std::setw(14) << "tolerance"
           << "\n";

        for (size_t i = 0; i < result.worstMismatches.size(); ++i) {
            const auto& m = result.worstMismatches[i];
            ss << "    " << std::left << std::setw(3) << (i + 1) << std::setw(static_cast<int>(maxIndexWidth + 2))
               << formatIndex(m.multiIndex) << std::right << std::setprecision(6) << std::setw(14) << m.expected
               << std::setw(14) << m.actual << std::setw(14) << m.absError << std::setw(14) << m.tolerance << "\n";
        }
    }

    return ss.str();
}

std::string formatExpectedAnomalyWarning(const TensorComparisonResult& result, std::string_view tensorName) {
    if (!result.hasExpectedAnomalies()) {
        return {};
    }
    std::ostringstream ss;
    ss << "WARNING: Expected tensor";
    if (!tensorName.empty()) {
        ss << " \"" << tensorName << "\"";
    }
    ss << " contains";
    if (result.expectedNanCount > 0) {
        ss << " " << result.expectedNanCount << " NaN";
    }
    if (result.expectedNanCount > 0 && result.expectedInfCount > 0) {
        ss << " and";
    }
    if (result.expectedInfCount > 0) {
        ss << " " << result.expectedInfCount << " Inf";
    }
    ss << " value(s) -- reference data may be corrupt";
    return ss.str();
}

void assertTensorsClose(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual,
                        double absThreshold, double relThreshold, size_t topN) {
    ASSERT_EQ(expected.size(), actual.size()) << "Number of output tensors mismatch";

    std::ostringstream failures;
    size_t failureCount = 0;

    for (size_t i = 0; i < expected.size(); ++i) {
        const auto name = "output_" + std::to_string(i);
        auto result = compareTensors(expected[i], actual[i], absThreshold, relThreshold, topN);

        if (!result.passed()) {
            ++failureCount;
            failures << "\n" << formatComparisonResult(result, name);
        }
    }

    ASSERT_EQ(failureCount, 0u) << "Tensor comparison failed for " << failureCount << " output(s):" << failures.str();
}

}  // namespace ov::test::utils
