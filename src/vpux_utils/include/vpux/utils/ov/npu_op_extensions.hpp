//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/extension.hpp>

#include <vector>

namespace vpux {

/// \brief Returns the custom op extensions needed so an ov::Core (or the compiler's own
/// model deserializer) can construct NPU-specific ops (DynamicQuantize, RMS, RoPE,
/// GroupQueryAttention, GatedDeltaNet, FlashAttentionTile) when reading an IR model that
/// was serialized with them. Shared between the compiler frontend (model_preprocessor.cpp)
/// and standalone tools (e.g. infer_tool) so the op list stays in one place.
std::vector<ov::Extension::Ptr> getNpuOpExtensions();

}  // namespace vpux
