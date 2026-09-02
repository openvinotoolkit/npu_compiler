//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"

#include <llvm/ADT/Hashing.h>

namespace vpux::VPU::VF::v2 {

inline llvm::hash_code getShapeHash(ShapeRef shape) {
    return llvm::hash_combine_range(shape.begin(), shape.end());
}

inline llvm::hash_code getTileHash(const TileInfo& tile, bool hashShapeOnly = false) {
    auto hash = getShapeHash(tile.shape);
    if (hashShapeOnly) {
        return hash;
    }
    hash = llvm::hash_combine(hash, llvm::hash_combine_range(tile.offsets.begin(), tile.offsets.end()));
    hash = llvm::hash_combine(hash, llvm::hash_combine_range(tile.axis.begin(), tile.axis.end()));
    return llvm::hash_combine(hash, tile.isCompletedTile);
}

}  // namespace vpux::VPU::VF::v2
