//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdint>
#include <vector>

namespace vpux::bitc {
// clang-format off
enum class ArchType : uint32_t { UNKNOWN, NPU27, NPU4, NPU5
};

struct BitCompactorConfig {
    ArchType arch_type{ArchType::UNKNOWN};
    bool sparse_mode_enable{false};
    bool weight_compress_enable{true};
    bool bypass_compression{false};
    bool mode_fp16_enable{false};  // NPU4+ only

    // For sparse mode
    std::vector<uint8_t> bitmap;
    unsigned sparse_block_size{0};
};
// clang-format on
}  // namespace vpux::bitc

#include "Encoder.hpp"
