//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstdint>
#include <vector>

namespace vpux::bitc {
// clang-format off
enum class ArchType : uint32_t { NPU27, NPU4
    , NPU5
};

struct BitCompactorConfig {
    ArchType arch_type;
    bool sparse_mode_enable{false};  // NPU5 only
    bool weight_compress_enable{true};
    bool bypass_compression{false};
    bool mode_fp16_enable{false};  // NPU4+ only

    // For sparse mode
    std::vector<uint8_t> bitmap;
    unsigned sparse_block_size;
};
// clang-format on
}  // namespace vpux::bitc

#include "Encoder.hpp"
