//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/group_query_attention.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ov {
class Model;
}  // namespace ov

namespace vpux {
namespace config {
enum class ArchKind : uint64_t;
}  // namespace config
namespace IE {

// Returns whether the given optional GroupQueryAttention input is really provided: absent optional inputs
// are padded with empty-constant placeholders (mirrors `has_input` in GroupQueryAttentionDecomposition).
bool hasGroupQueryAttentionInput(const std::shared_ptr<const ov::op::internal::GroupQueryAttention>& origNode,
                                 ov::op::internal::GroupQueryAttentionInputs input);

// A GQA node with a seqlens_k input and a statically known total sequence length greater than this
// threshold always becomes an IE::AttentionDMAOp running the attention_dma_flash kernel (seqlens_k
// unconditionally selects that kernel in sw_kernel.cpp), so decomposing it first only wastes compile
// time and schedule quality for no benefit.
constexpr int64_t GQA_FLASH_KERNEL_MIN_SEQ_LEN = 512;

// Returns the total sequence length a GroupQueryAttention node is configured for, when statically known.
//
// past_key has shape [batch, kv_num_heads, cache_seq_len, head_size]; `cache_seq_len` depends on the caching
// mode:
//  - Sliding-window cache: the window capacity. The true total is unbounded, but the decomposed subgraph is
//    bounded by the capacity, so it is used as-is.
//  - Full-length cache (the common static-shape NPU case): already the total (past + current) length, since
//    the decomposition writes the current query/key in place via ScatterUpdate. Only holds when the current
//    chunk length is static too; otherwise the cache grows at runtime and past_key holds only the past.
//
// past_key/past_value are optional in ONNX but mandatory on the internal op, so the size/rank guards in the
// implementation are defensive only.
std::optional<int64_t> getGQATotalSequenceLength(
        const std::shared_ptr<const ov::op::internal::GroupQueryAttention>& gqa);

// Returns true when the GQA node is supported by the AttentionDMA flash kernel, so OV decomposition is skipped for it.
// Decided per node: causal, seqlens_k present, static total sequence length >= GQA_FLASH_KERNEL_MIN_SEQ_LEN,
// 16-aligned dims, full-head RoPE and an 8-bit i8 quantized KV cache.
bool isGQAFlashEligible(const std::shared_ptr<const ov::Node>& node, int optimizationLevel, config::ArchKind arch);

}  // namespace IE
}  // namespace vpux
