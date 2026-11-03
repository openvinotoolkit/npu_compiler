//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/ov/npu_op_extensions.hpp"

#include <intel_npu/ops/flash_attention_tile.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/op/gated_delta_net.hpp>
#include <openvino/op/group_query_attention.hpp>
#include <ov_ops/dynamic_quantize.hpp>
#include <ov_ops/rms.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

namespace vpux {

std::vector<ov::Extension::Ptr> getNpuOpExtensions() {
    return {std::make_shared<ov::OpExtension<ov::op::internal::DynamicQuantize>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::RMS>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::RoPE>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::GroupQueryAttention>>(),
            std::make_shared<ov::OpExtension<ov::op::internal::GatedDeltaNet>>(),
            std::make_shared<ov::OpExtension<ov::intel_npu::op::FlashAttentionTile>>()};
}

}  // namespace vpux
