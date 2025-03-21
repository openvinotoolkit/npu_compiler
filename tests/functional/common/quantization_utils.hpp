//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/core.hpp>

#include <tuple>
#include <vector>

namespace ov::test {

class FakeQuantizeParams {
public:
    const std::vector<float> inLow, inHigh, outLow, outHigh;

    FakeQuantizeParams(std::vector<float>&& inLow, std::vector<float>&& inHigh, std::vector<float>&& outLow,
                       std::vector<float>&& outHigh) noexcept;
    FakeQuantizeParams(const FakeQuantizeParams&) = default;
    FakeQuantizeParams(FakeQuantizeParams&&) noexcept = default;
    virtual ~FakeQuantizeParams() noexcept = default;
};

std::ostream& operator<<(std::ostream& os, const FakeQuantizeParams& params);

namespace utils {

// Creates a FakeQuantize from input/output quantization intervals.
// The input and output intervals can have different number of channels (not possible with current OV versions).
std::shared_ptr<ov::Node> makeFakeQuantize(const ov::Output<ov::Node>& in, const ov::element::Type& type,
                                           std::size_t levels, const FakeQuantizeParams& params);

}  // namespace utils
}  // namespace ov::test
