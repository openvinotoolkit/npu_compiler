//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"

#pragma once

namespace vpux {
namespace VPUIP {

struct Tile final {
    Tile() = delete;

    explicit Tile(size_t rank): shape(rank), offsets(rank) {
    }

    explicit Tile(ShapeRef shape): shape(shape.raw()), offsets(shape.size(), 0) {
    }

    Shape shape;
    Shape offsets;
};

struct PadsTileConfig final {
    SmallVector<int64_t> begin;
    SmallVector<int64_t> end;
};

struct ConvTileConfig final {
    Tile inputTile;
    Tile filterTile;
    Tile biasTile;
    PadsTileConfig pads;
};

class Tiling final {
public:
    // helper function to generate a set of tiles from dividing a shape. A shape divided across multiple dimensions will
    // generate a set of tiles, each having its own size and offsets
    static SmallVector<Tile> fillDividedTiles(ShapeRef divisors, ShapeRef orig);

    static PadsTileConfig backInferPadsTile(const Tile& outputTile, ShapeRef outShape, ArrayRef<int64_t> opPadsBegin,
                                            ArrayRef<int64_t> opPadsEnd);

    static ConvTileConfig backInferConvTile(IERT::ConvolutionOp origOp, const Tile& outputTile);
};

}  // namespace VPUIP
}  // namespace vpux
