//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/shave.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

int64_t vpux::getShaveQueueIdEncoding(int64_t numTiles, int64_t tileIndex, int64_t listIndex) {
    VPUX_THROW_UNLESS(numTiles > 0, "Incorrect number of tiles: {0}", numTiles);
    VPUX_THROW_UNLESS(tileIndex < numTiles, "Incorrect tile index ({0}) for given number of tiles ({1})", tileIndex,
                      numTiles);
    return listIndex * numTiles + tileIndex;
}

int64_t vpux::getShaveTileIndexFromEncodedId(int64_t shaveQueueIdEncoding, int64_t numTiles) {
    VPUX_THROW_UNLESS(numTiles > 0, "Incorrect number of tiles: {0}", numTiles);
    return shaveQueueIdEncoding % numTiles;
}

int64_t vpux::getShaveListIndexFromEncodedId(int64_t shaveQueueIdEncoding, int64_t numTiles) {
    VPUX_THROW_UNLESS(numTiles > 0, "Incorrect number of tiles: {0}", numTiles);
    return shaveQueueIdEncoding / numTiles;
}
