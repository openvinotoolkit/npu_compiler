//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/frontend/group_query_attention_utils.hpp"

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include <openvino/core/validation_util.hpp>

namespace vpux {
namespace IE {

namespace {

// Statically known innermost (head size) dim of a GQA input, or nullopt when the rank/dim is dynamic.
std::optional<int64_t> getGQAHeadDim(const std::shared_ptr<const ov::op::internal::GroupQueryAttention>& gqa,
                                     ov::op::internal::GroupQueryAttentionInputs input) {
    const auto pos = static_cast<size_t>(input);
    if (gqa->get_input_size() <= pos) {
        return std::nullopt;
    }
    const auto& shape = gqa->get_input_partial_shape(pos);
    if (shape.rank().is_dynamic() || shape.rank().get_length() != 4 || !shape[3].is_static()) {
        return std::nullopt;
    }
    return shape[3].get_length();
}

// Whether the target supports the AttentionDMA flash kernel.
bool isPlatformCompatible(config::ArchKind arch, int optimizationLevel) {
    constexpr int kAttentionDmaFlashOptimizationLevel = 3;
    switch (arch) {
    case config::ArchKind::NPU40XX:
        return optimizationLevel == kAttentionDmaFlashOptimizationLevel;
    case config::ArchKind::NPU50XX:
        return true;
    case config::ArchKind::NPU60XX:
        return true;
    default:
        return false;
    }
}

}  // namespace

std::optional<int64_t> getGQATotalSequenceLength(
        const std::shared_ptr<const ov::op::internal::GroupQueryAttention>& gqa) {
    constexpr size_t pastKeyInputIdx = static_cast<size_t>(ov::op::internal::GroupQueryAttentionInputs::PAST_KEY);
    if (gqa->get_input_size() <= pastKeyInputIdx) {
        return std::nullopt;
    }

    const auto& pastKeyShape = gqa->get_input_partial_shape(pastKeyInputIdx);
    if (pastKeyShape.rank().is_dynamic() || pastKeyShape.rank().get_length() != 4) {
        return std::nullopt;
    }

    const auto& cacheSeqLenDim = pastKeyShape[2];
    if (!cacheSeqLenDim.is_static()) {
        return std::nullopt;
    }

    if (gqa->get_sliding_window_cache()) {
        return cacheSeqLenDim.get_length();
    }

    constexpr size_t queryInputIdx = static_cast<size_t>(ov::op::internal::GroupQueryAttentionInputs::QUERY);
    if (gqa->get_input_size() <= queryInputIdx) {
        return std::nullopt;
    }

    const auto& queryShape = gqa->get_input_partial_shape(queryInputIdx);
    if (queryShape.rank().is_dynamic() || queryShape.rank().get_length() != 4) {
        return std::nullopt;
    }

    const auto& currentSeqLenDim = queryShape[2];
    if (!currentSeqLenDim.is_static()) {
        return std::nullopt;
    }

    return cacheSeqLenDim.get_length();
}

bool isGQAFlashEligible(const std::shared_ptr<const ov::Node>& node, int optimizationLevel, config::ArchKind arch) {
    using GQAInput = ov::op::internal::GroupQueryAttentionInputs;

    if (!isPlatformCompatible(arch, optimizationLevel)) {
        return false;
    }

    auto gqa = ov::as_type_ptr<const ov::op::internal::GroupQueryAttention>(node);
    if (gqa == nullptr || !gqa->get_causal() || !hasGroupQueryAttentionInput(gqa, GQAInput::SEQLENS_K)) {
        return false;
    }

    const auto totalSeqLen = getGQATotalSequenceLength(gqa);
    const auto eDim = getGQAHeadDim(gqa, GQAInput::QUERY);
    const auto eVDim = getGQAHeadDim(gqa, GQAInput::VALUE);
    if (!totalSeqLen.has_value() || !eDim.has_value() || !eVDim.has_value() ||
        *totalSeqLen < GQA_FLASH_KERNEL_MIN_SEQ_LEN || *totalSeqLen % 16 != 0 || *eDim % 16 != 0 || *eVDim % 16 != 0) {
        return false;
    }

    // RoPE must cover the full head.
    if (gqa->get_do_rotary()) {
        if (!hasGroupQueryAttentionInput(gqa, GQAInput::COS_CACHE)) {
            return false;
        }
        const auto& cosShape = gqa->get_input_partial_shape(static_cast<size_t>(GQAInput::COS_CACHE));
        if (cosShape.rank().is_dynamic() || cosShape.rank().get_length() != 2 || !cosShape[1].is_static() ||
            2 * cosShape[1].get_length() != *eDim) {
            return false;
        }
    }

    // A quantized KV cache must be 8-bit i8.
    if (gqa->is_kv_quantized()) {
        return gqa->get_kv_cache_bit_width() == 8 &&
               gqa->get_input_element_type(static_cast<size_t>(GQAInput::PAST_KEY)) == ov::element::i8;
    }

    return true;
}

bool hasGroupQueryAttentionInput(const std::shared_ptr<const ov::op::internal::GroupQueryAttention>& origNode,
                                 ov::op::internal::GroupQueryAttentionInputs input) {
    const auto pos = static_cast<size_t>(input);
    return pos < origNode->get_input_size() && !ov::util::is_empty_constant_tensor(origNode->input_value(pos));
}

}  // namespace IE
}  // namespace vpux
