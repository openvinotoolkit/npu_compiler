//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "quantization_utils.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "vpux/utils/core/error.hpp"

ov::test::FakeQuantizeParams::FakeQuantizeParams(std::vector<float>&& inLow, std::vector<float>&& inHigh,
                                                 std::vector<float>&& outLow, std::vector<float>&& outHigh) noexcept
        : inLow(std::move(inLow)), inHigh(std::move(inHigh)), outLow(std::move(outLow)), outHigh(std::move(outHigh)) {
    VPUX_THROW_WHEN(this->inLow.size() != this->inHigh.size() || this->outLow.size() != this->outHigh.size(),
                    "Different FakeQuantize low/high sizes.");
}

std::ostream& ov::test::operator<<(std::ostream& os, const FakeQuantizeParams& params) {
    os << "{iL=" << utils::vec2str(params.inLow) << ".iH=" << utils::vec2str(params.inHigh)
       << ".oL=" << utils::vec2str(params.outLow) << ".oH=" << utils::vec2str(params.outHigh) << "}";
    return os;
}

std::shared_ptr<ov::Node> ov::test::utils::makeFakeQuantize(const ov::Output<ov::Node>& in,
                                                            const ov::element::Type& type, std::size_t levels,
                                                            const FakeQuantizeParams& params) {
    auto inputLowNode = ov::test::utils::make_constant(type, {1, params.inLow.size(), 1, 1}, params.inLow);
    auto inputHighNode = ov::test::utils::make_constant(type, {1, params.inHigh.size(), 1, 1}, params.inHigh);
    auto outputLowNode = ov::test::utils::make_constant(type, {1, params.outLow.size(), 1, 1}, params.outLow);
    auto outputHighNode = ov::test::utils::make_constant(type, {1, params.outHigh.size(), 1, 1}, params.outHigh);

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(in, inputLowNode, inputHighNode, outputLowNode, outputHighNode,
                                                         levels);
    return fq;
}
