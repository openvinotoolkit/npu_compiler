//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace vpux {

enum class LoopType {
    None,
    Tiling,
    VF,
};

inline const char* toString(LoopType type) {
    switch (type) {
    case LoopType::None:
        return "None";
    case LoopType::Tiling:
        return "Tiling";
    case LoopType::VF:
        return "VF";
    default:
        return "Unknown";
    }
}

}  // namespace vpux
