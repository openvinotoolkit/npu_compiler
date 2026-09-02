//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>

namespace vpux {
namespace IE {

// Legal Attention configurations: {qHeadSize, tSL, sSL}
struct AttentionConfig {
    int64_t qHeadSize;
    int64_t tSL;
    int64_t sSL;
};

inline constexpr std::array<AttentionConfig, 18> LEGAL_ATTENTION_CONFIGS = {{{192, 225, 225},
                                                                             {12, 3600, 3600},
                                                                             {8, 300, 300},
                                                                             {16, 577, 577},
                                                                             {10, 1024, 1024},
                                                                             {10, 1024, 77},
                                                                             {20, 256, 256},
                                                                             {20, 256, 77},
                                                                             {6, 3072, 3072},
                                                                             {6, 151, 151},
                                                                             {12, 512, 512},
                                                                             {16, 256, 256},
                                                                             {1, 80, 826},
                                                                             {6, 2752, 2752},
                                                                             {6, 1886, 1886},
                                                                             {20, 1500, 1500},
                                                                             {32, 1200, 1200},
                                                                             {32, 1200, 160}}};

}  // namespace IE
}  // namespace vpux
